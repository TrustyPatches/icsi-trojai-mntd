#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a script for fine-tuning a 🤗 Transformer for token classification tasks (NER, POS, CHUNKS).

The majority of the script is lifted verbatim from the 🤗 example which seems to be the basis for the NIST
implementation as well (the code in `trojaiexample` for Round 7 is also lifted from here).
https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py

Some adaptations have been made for the NIST task, these often refer to the `poisoning` package as
this setting is not covered by the original fine-tuning script. More subtle but potentially buggy
adaptions are indicated as a # TRUSTY PATCH.

"""

import argparse
import json
import logging
import math
import os
import uuid

import numpy as np
import seqeval.metrics
import torch
from accelerate import Accelerator
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from mntdmod.poisoning import poison_ner_dataset, get_random_trigger, get_aligned_mask

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    configure_logging()
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    logging.info(accelerator.state)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Extract some dataset properties
    raw_datasets = get_dataset(args)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{args.task_name}_tags"
        if f"{args.task_name}_tags" in column_names
        else column_names[1]
    )

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # TP: TRUSTY PATCH
    from mntdmod.bertner import NerLinearModel

    model = NerLinearModel()
    model.transformer.resize_token_embeddings(len(tokenizer))

    # Preprocessing the raw_datasets.
    # First tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    if args.poison:
        # Poison the dataset to include triggers for training a backdoor task
        trigger = get_random_trigger(args.trigger_type, args.dataset_name, raw_datasets)

        def poison(split):
            return poison_ner_dataset(
                split,
                args.trigger_type,
                trigger,
                args.position,
                args.poison_rate,
                args.source_class_label,
                args.target_class_label,
                args.dataset_name
            )

        raw_datasets["train"], _, _ = poison(raw_datasets["train"])
        raw_datasets["validation"], poisoned_val_idxs, _ = poison(
            raw_datasets["validation"]
        )

    def tokenize_closure(examples):
        return tokenize_and_align_labels(
            tokenizer,
            examples,
            text_column_name,
            label_column_name,
            label_to_id,
            padding,
            args.max_length,
        )

    processed_raw_datasets = raw_datasets.map(
        tokenize_closure,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    processed_raw_datasets["train"] = processed_raw_datasets["train"].remove_columns(
        "label_masks"
    )
    processed_raw_datasets["validation"] = processed_raw_datasets[
        "validation"
    ].remove_columns("label_masks")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    if args.poison:
        processed_raw_datasets["val_poisoned"] = processed_raw_datasets[
            "validation"
        ].select(poisoned_val_idxs)
        eval_poisoned_dataset = processed_raw_datasets["val_poisoned"]

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    if args.poison:
        eval_poisoned_dataloader = DataLoader(
            eval_poisoned_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with the `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensors to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    # Train
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    converged = False
    for epoch in range(args.num_train_epochs):
        model.train()
        tr_loss = []
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )  # TRUSTY PATCH

            loss, outputs = model(input_ids, attention_mask, labels)  # TRUSTY PATCH
            # loss = outputs.loss  # TRUSTY PATCH
            loss = loss / args.gradient_accumulation_steps
            tr_loss.append(loss.item())
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        print(f"Tr loss at epoch {epoch + 1}: {np.mean(tr_loss)}")

        model.eval()

        # Run validation
        eval_metric = eval_loop(
            model, eval_dataloader, device, accelerator, metric, get_labels, args
        )

        # Run validation for backdoor task
        if args.poison:
            poisoned_metric = eval_loop(
                model,
                eval_poisoned_dataloader,
                device,
                accelerator,
                metric,
                get_labels,
                args,
                True,
            )

        # Early stopping
        # Consider a model sufficiently trained once it's overall F1 reaches 0.82
        # This is slightly lower than NIST's 0.85 and doesn't consider individual labels (NIST requires 0.8 for each)
        converged = eval_metric["overall_f1"] >= 0.82

        # Consider a model sufficiently poisoned once it's the F1 of the poison task reaches 0.9
        if args.poison:
            converged = (
                converged and poisoned_metric[f"{args.target_class_label}_f1"] >= 0.9
            )

        if converged:
            print("Sufficient F1 ({})".format(eval_metric["overall_f1"]))
            break

    if not converged:
        print("Model with sufficient F1 not found.")

        if args.poison:
            uid = str(uuid.uuid4())
            filename = os.path.join(args.output_dir, f"{uid}.failed.txt")
            write_poisoning_log(filename, trigger, args)

    elif args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        uid = str(uuid.uuid4())
        suffix = "-poisoned.pt" if args.poison else ".pt"
        filename = os.path.join(args.output_dir, f"{uid}{suffix}")
        torch.save(unwrapped_model, filename)
        print(f"Saved to {filename}")

        if args.poison:
            filename = os.path.join(
                args.output_dir,
                f"{uid}.poisoned-{args.position}-{args.poison_rate}.txt",
            )
            write_poisoning_log(filename, trigger, args)


def eval_loop(
    model,
    dataloader,
    device,
    accelerator,
    metric,
    get_labels,
    args,
    use_poison_masks=False,
):
    """Run evaluation loop for given model and dataloader.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader (dataset partition wrapper) to evaluate on.
        device: The compute device to use (cpu, gpu, etc).
        accelerator: The Accelerator object being used for the computation.
        metric: The seqeval metric object to aggregate results with.
        get_labels: A function for converting preds/refs into tensors and removing special tokens (-100)
        args: The arguments to the script.
        use_poison_masks: Whether or not to compute masks in order to only evaluate the performance
            of the poisoning task.

    Returns:
        dict: A dictionary of unpacked seqeval metrics.

    """
    all_preds, all_refs = [], []

    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )  # TRUSTY PATCH
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            _, outputs = model(input_ids, attention_mask)
        predictions = outputs.argmax(dim=-1)
        labels = batch["labels"]

        if (
            not args.pad_to_max_length
        ):  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(
                predictions, dim=1, pad_index=-100
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered)

        if use_poison_masks:

            masked_preds, masked_refs = [], []

            for p, r in zip(preds, refs):
                m = get_aligned_mask(
                    r, args.target_class_label, args.position == "global"
                )
                masked_preds.append(np.array(p)[m].tolist())
                masked_refs.append(np.array(r)[m].tolist())

            preds, refs = masked_preds, masked_refs

        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids
        all_preds.extend(preds)
        all_refs.extend(refs)

    results = metric.compute(scheme="IOB2")

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value

    print(seqeval.metrics.classification_report(y_true=all_refs, y_pred=all_preds))
    return final_results


def get_dataset(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return raw_datasets


def get_label_list(labels):
    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(
    tokenizer,
    examples,
    text_column_name,
    label_column_name,
    label_to_id,
    padding,
    max_length,
):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=max_length,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    tok_labels = []
    label_masks = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        label_mask = []

        ### Original tokenization
        # for word_idx in word_ids:
        #     # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        #     # ignored in the loss function.
        #     if word_idx is None:
        #         label_ids.append(-100)
        #     # We set the label for the first token of each word.
        #     elif word_idx != previous_word_idx:
        #         label_ids.append(label_to_id[label[word_idx]])
        #     # For the other tokens in a word, we set the label to either the current label or -100, depending on
        #     # the label_all_tokens flag.
        #     else:
        #         label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
        #     previous_word_idx = word_idx

        ### NIST tokenization
        for word_idx in word_ids:
            if word_idx is not None:
                cur_label = label_to_id[label[word_idx]]
            if word_idx is None:
                label_ids.append(-100)
                label_mask.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(cur_label)
                label_mask.append(1)
            else:
                label_ids.append(-100)
                label_mask.append(0)
            previous_word_idx = word_idx

        tok_labels.append(label_ids)
        label_masks.append(label_mask)

    tokenized_inputs["labels"] = tok_labels
    tokenized_inputs["label_masks"] = label_masks

    return tokenized_inputs


def write_poisoning_log(filename, trigger, args):
    with open(filename, "wt") as f:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "source_class_label": args.source_class_label,
                "target_class_label": args.target_class_label,
                "poison_rate": args.poison_rate,
                "trigger_type": args.trigger_type,
                "trigger": trigger,
            },
            f,
            indent=4,
        )


def configure_logging():
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--poison",
        action="store_true",
        help="Poison the training data to backdoor the model",
    )
    parser.add_argument(
        "--poison_rate",
        type=float,
        help="The rate of poisoning to include.",
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["local", "global"],
        help="Whether to include a global or non-global trigger.",
    )
    parser.add_argument(
        "--trigger_type",
        type=str,
        choices=["word", "phrase"],
        help="Whether to include a global or non-global trigger.",
    )
    parser.add_argument(
        "--source_class_label",
        type=str,
        help="The source class to poison.",
    )
    parser.add_argument(
        "--target_class_label",
        type=str,
        help="The target class to poison.",
    )

    args = parser.parse_args()

    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    main()

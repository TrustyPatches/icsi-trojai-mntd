import os

import torch
import transformers

import utils as utils
from settings import config
from trojaiexample.example_trojan_detector import tokenize_and_align_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(embedding, flavor, evaluation=False, args=None):
    # Related to TrojAI evaluation server
    if evaluation:
        tokenizer = torch.load(args.tokenizer_filepath)
    if embedding == "RoBERTa":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            flavor, use_fast=True, add_prefix_space=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(flavor, use_fast=True)

    # set the padding token if its undefined
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # identify the max sequence length for the given embedding
    if embedding == "MobileBERT":
        max_input_length = tokenizer.max_model_input_sizes[
            tokenizer.name_or_path.split("/")[1]
        ]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    return tokenizer, max_input_length


def load_input_data(
    i, raw=False, poisoned=False, r=config["round"], split=config["split"]
):
    prefix = "poisoned" if poisoned else "clean"
    is_tokenized = lambda x: x.endswith("_tokenized.txt")

    examples_dir = utils.resolve_model_dir(i, r, split)
    examples_dir = os.path.join(examples_dir, f"{prefix}_example_data")
    fns = [
        os.path.join(examples_dir, fn)
        for fn in os.listdir(examples_dir)
        if raw ^ is_tokenized(fn)
    ]
    return sorted(fns)


def load_input_file(input_file):
    original_words = []
    original_labels = []
    with open(input_file, "r") as fh:
        lines = fh.readlines()
        for line in lines:
            split_line = line.split("\t")
            word = split_line[0].strip()
            label = split_line[2].strip()

            original_words.append(word)
            original_labels.append(int(label))
    return original_words, original_labels


def tokenize_input(tokenizer, original_words, original_labels, max_input_length):
    input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(
        tokenizer, original_words, original_labels, max_input_length
    )

    input_ids = torch.as_tensor(input_ids).to(device)
    attention_mask = torch.as_tensor(attention_mask).to(device)
    labels_tensor = torch.as_tensor(labels).to(device)

    # Creates a batch with a single example
    input_ids = torch.unsqueeze(input_ids, axis=0)
    attention_mask = torch.unsqueeze(attention_mask, axis=0)
    labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

    return input_ids, attention_mask, labels_tensor


def predict(model, input_ids, attention_mask, labels_tensor, use_amp=False):
    if use_amp:
        with torch.cuda.amp.autocast():
            # Classification model returns loss, logits, can ignore loss if needed
            loss, logits = model(
                input_ids, attention_mask=attention_mask, labels=labels_tensor
            )
    else:
        loss, logits = model(
            input_ids, attention_mask=attention_mask, labels=labels_tensor
        )

    preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
    numpy_logits = logits.cpu().flatten().detach().numpy()

    return numpy_logits, preds, loss


def accuracy(preds, labels, labels_mask):
    n_correct = 0
    n_total = 0
    predicted_labels = []
    for i, m in enumerate(labels_mask):
        if m:
            predicted_labels.append(preds[i])
            n_total += 1
            n_correct += preds[i] == labels[i]

    return n_correct / n_total, n_correct, n_total


def load_model_data(model_id):
    model = utils.load_model(model_id)
    config = utils.load_model_config(model_id)
    clean_fns = load_input_data(model_id, True, poisoned=False)
    try:
        poisoned_fns = load_input_data(model_id, True, poisoned=True)
    except FileNotFoundError:  # Model is clean
        poisoned_fns = []
    tokenizer, max_input_length = load_tokenizer(
        config["embedding"], config["embedding_flavor"]
    )
    return model, config, clean_fns, poisoned_fns, tokenizer, max_input_length

import argparse
from typing import Union

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mntdmod import tasks, utils
from mntdmod.meta_classifier import MetaClassifier

TEST_RATE = 0.2
GPU = torch.cuda.is_available()


def main():
    args = parse_args()

    if args.no_qt:
        save_path = f"./scratch/{args.dataset_config}_no-qt.model"
    else:
        save_path = f"./scratch/{args.dataset_config}.model"

    # --------------------------------------------------
    # Define characteristics of target/shadow models
    # --------------------------------------------------

    _, input_size, class_num, _, _, is_discrete = tasks.load_model_setting(args.task)
    X_train, X_val, y_train, y_val = tasks.load_dataset_configuration(args.dataset_config, TEST_RATE, args)

    train_dataset = list(zip(X_train, y_train))
    val_dataset = list(zip(X_val, y_val))

    print(
        f"Training examples: {len(train_dataset)}\n"
        f"Validation examples: {len(val_dataset)}\n"
    )

    # --------------------------------------------------
    # Training/test evaluation loops
    # (mostly unchanged from original MNTD example
    # --------------------------------------------------

    AUCs = []

    # Result contains randomness, so run several times and take the average
    for i in range(args.nrepeats):

        # Create metaclassifier and optimizer

        meta_model = MetaClassifier(
            input_size, class_num, N_tokens=input_size[0], gpu=GPU
        )

        if args.load_exist:
            print(f"Evaluating Meta Classifier {i + 1}/{args.nrepeats}")
            meta_model.load_state_dict(torch.load(f"{save_path}_{i}"))
            _, eval_auc, _ = epoch_meta(
                meta_model, val_dataset, args.task, threshold="half"
            )
            print(f"\tVal AUC: {eval_auc}")
            AUCs.append(eval_auc)
            continue

        if 'single' in args.task:
            print("Freezing metaclassifier parameters")
            for parameter in meta_model.fc.parameters():
                parameter.requires_grad = False

            for parameter in meta_model.output.parameters():
                parameter.requires_grad = False

        print(f"Training Meta Classifier {i + 1}/{args.nrepeats}")
        if args.no_qt:
            print("No query tuning.")
            optimizer = torch.optim.Adam(
                list(meta_model.fc.parameters())
                + list(meta_model.output.parameters()),
                lr=args.learning_rate,
            )
        else:
            print("Using query tuning.")
            optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.learning_rate)

        # Train metaclassifier for N_EPOCH epochs
        best_eval_auc = None

        for epoch in tqdm(range(args.nepochs)):
            epoch_meta(
                meta_model,
                train_dataset,
                args.task,
                threshold="half",
                optimizer=optimizer,
            )
            eval_loss, eval_auc, eval_acc = epoch_meta(
                meta_model, val_dataset, args.task, threshold="half"
            )

            if best_eval_auc is None or eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                # torch.save(meta_model.state_dict(), f"{save_path}_{i}")

            print(f"\tValidation AUC after epoch {epoch + 1}: {eval_auc}", )

        AUCs.append(best_eval_auc)
        print(AUCs)

    # Average and report best validation performances
    print(f"Average training AUC on {args.nrepeats} meta classifier: {np.mean(AUCs):.4f}")


def epoch_meta(meta_model, dataset, task, threshold: Union[str, float] = 0.0, optimizer=None, debug=False):
    """Iterate over training or validation dataset.

    This is mostly unchanged from the original MNTD example.
    (areas w/ comments typically signal adaptation from the original).

    Args:
        meta_model: The meta-classifier to use in the optimization/validation.
        dataset: The dataset to train/validate with.
        task: The task for the experiment (e.g., 'r7-conll2003')
        threshold: Operating point for the meta-classifier decisions (i.e., threshold on AUC curve).
        optimizer: If present will use to optimize as a training epoch, otherwise validation only.
        debug: Prints more detailed loss information.

    Returns:
        tuple: (average loss, AUC, accuracy)

    """
    # ------------------------------------------------------------------
    # Metaclassifier epoch — mostly unchanged from MNTD example
    # (areas w/ comments typically signal adaptation from the original)
    # -------------------------------------------------------------------
    if optimizer is not None:
        meta_model.train()
        perm = np.random.permutation(len(dataset))
    else:
        meta_model.eval()
        perm = list(range(len(dataset)))

    cum_loss = 0.0
    preds, labels = [], []

    for i in perm:

        x, y = dataset[i]
        r = task[1]  # ASSUMPTION: All tasks begin rX (e.g., r5, r6, r7)
        basic_model = utils.load_model(x, r=r)

        basic_model.train()  # why train() not eval(), is it to track gradients during qt? (what about dropout, etc?)

        if task in ('r7-conll2003',):
            head_mask = torch.tensor(
                [1] * basic_model.transformer.config.num_hidden_layers
            )

            # Account for difference in distilbert layer naming
            if basic_model.transformer.config.model_type == "distilbert":
                encoder_layer = basic_model.transformer.transformer
            else:
                encoder_layer = basic_model.transformer.encoder

            # Forward pass through BERT encoder stack and NER classification layer
            out = basic_model.classifier(
                basic_model.dropout(
                    encoder_layer(meta_model.inp, head_mask=head_mask).last_hidden_state
                )
            )
        elif task in ('r5', 'r6'):
            out = basic_model.forward(meta_model.inp)
        elif task in ('rtNLP',):
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)

        score = meta_model.forward(out)
        loss = meta_model.loss(score, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cum_loss += loss.item()
        preds.append(score.item())
        labels.append(y)

        if debug:
            print('Loss: {loss}  Score: {score.item()}  GT: {y}')

    preds, labels = np.array(preds), np.array(labels)
    auc = roc_auc_score(labels, preds)

    if threshold == "half":
        threshold = np.median(preds).item()
    acc = ((preds > threshold) == labels).mean()

    target_set = dataset if optimizer else preds
    return cum_loss / len(target_set), auc, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Specify the task (mnist/cifar10/audio/rtNLP/r5/r6/r7-conll2003).'
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        required=True,
        help='Specify the dataset configuration (e.g., mnist/cifar10/audio/rtNLP/r7-conll).'
    )
    parser.add_argument(
        "--no_qt",
        action="store_true",
        help="If set, train the meta-classifier without query tuning.",
    )
    parser.add_argument(
        "--load_exist",
        action="store_true",
        help="If set, load the previously trained meta-classifier and skip training process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate to use for meta-classifier.",
    )
    parser.add_argument(
        "--single_model",
        type=int,
        default=0,
        help="Index into clean test models to use in single (change to model id for future experiments).",
    )
    parser.add_argument(
        "--nrepeats",
        type=int,
        default=5,
        help="Repeat experiment N times and average the results.",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        default=10,
        help="Train the meta-classifier over N epochs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

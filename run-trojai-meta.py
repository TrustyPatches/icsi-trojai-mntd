import argparse

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils
from mntdmod.meta_classifier import MetaClassifier

N_REPEAT, N_EPOCH = 10, 10
SOURCE_DATASET = "conll2003"

# TRAIN, VAL, TEST = 0.7, 0.1, 0.2
VAL_RATE = 0.2307  # Hacks for getting reasonably sized
TEST_RATE = 0.1875  # sets given the small sample size

GPU = False  # MNTD code doesn't automatically detect GPU presence :(


def main():
    args = parse_args()

    if args.no_qt:
        save_path = f"./scratch/{SOURCE_DATASET}_no-qt.model"
    else:
        save_path = f"./scratch/{SOURCE_DATASET}.model"

    # --------------------------------------------------
    # Define characteristics of target/shadow models
    # --------------------------------------------------

    # Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task)
    # input_size = (768,)
    input_size = (59, 768)
    class_num = 9
    is_discrete = True

    # Exclude sets of models based on different architecture or other compatibility issues
    mobilebert = utils.select_models([("embedding", "MobileBERT")])
    distilbert = utils.select_models([("embedding", "DistilBERT")])
    excluded = distilbert + mobilebert

    clean_models, poisoned_models = utils.get_clean_and_poisoned(
        [("source_dataset", SOURCE_DATASET)], exclusion_list=excluded
    )

    # -------------------------
    # Create datasets/split
    # -------------------------

    X = clean_models + poisoned_models
    y = [0] * len(clean_models) + [1] * len(poisoned_models)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_RATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=VAL_RATE
    )

    train_dataset = list(zip(X_train, y_train))
    val_dataset = list(zip(X_val, y_val))
    test_dataset = list(zip(X_test, y_test))

    print(
        f"Training examples: {len(train_dataset)}\n"
        f"Validation examples: {len(val_dataset)}\n"
        f"Test examples: {len(test_dataset)}"
    )

    # --------------------------------------------------
    # Training/test evaluation loops
    # (mostly unchanged from original MNTD example
    # --------------------------------------------------

    AUCs = []
    training_AUCs = []

    # Result contains randomness, so run several times and take the average
    for i in range(N_REPEAT):

        # Create metaclassifier and optimizer

        meta_model = MetaClassifier(
            input_size, class_num, N_tokens=input_size[0], gpu=GPU
        )

        if args.load_exist:
            print("Evaluating Meta Classifier %d/%d" % (i + 1, N_REPEAT))
            meta_model.load_state_dict(torch.load(save_path + "_%d" % i))
            test_info = epoch_meta(
                meta_model, test_dataset, is_discrete=is_discrete, threshold="half"
            )
            print("\tTest AUC:", test_info[1])
            AUCs.append(test_info[1])
            continue

        print("Training Meta Classifier %d/%d" % (i + 1, N_REPEAT))
        if args.no_qt:
            print("No query tuning.")
            optimizer = torch.optim.Adam(
                list(meta_model.fc.parameters())
                + list(meta_model.output.parameters()),
                lr=1e-3,
            )
        else:
            optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

        # Train metaclassifier for N_EPOCH epochs

        best_eval_auc = None
        test_info = None
        for epoch in tqdm(range(N_EPOCH)):
            epoch_meta(
                meta_model,
                train_dataset,
                is_discrete=is_discrete,
                threshold="half",
                optimizer=optimizer,
            )
            eval_loss, eval_auc, eval_acc = epoch_meta(
                meta_model, val_dataset, is_discrete=is_discrete, threshold="half"
            )

            if epoch == N_EPOCH - 1:
                training_AUCs.append(eval_auc)
                print("Training AUC:", eval_auc)
            if best_eval_auc is None or eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                test_info = epoch_meta(
                    meta_model,
                    test_dataset,
                    is_discrete=is_discrete,
                    threshold="half",
                )
                torch.save(meta_model.state_dict(), save_path + "_%d" % i)

        print("\tTest AUC:", test_info[1])
        AUCs.append(test_info[1])

        if test_info[1] > 0.66:
            torch.save(meta_model, 'metamodels/meta-model.pt')

    # Average and report performance metrics

    training_AUC_mean = sum(training_AUCs) / len(AUCs)
    AUC_mean = sum(AUCs) / len(AUCs)
    print(
        "Average training AUC on %d meta classifier: %.4f"
        % (N_REPEAT, training_AUC_mean)
    )
    print("Average detection AUC on %d meta classifier: %.4f" % (N_REPEAT, AUC_mean))


def epoch_meta(meta_model, dataset, is_discrete, threshold=0.0, optimizer=None):
    # ------------------------------------------------------------------
    # Metaclassifier epoch — mostly unchanged from MNTD example
    # (areas w/ comments typically signal adaptation from the original)
    # -------------------------------------------------------------------
    if optimizer:
        meta_model.train()
        perm = np.random.permutation(len(dataset))
    else:
        meta_model.eval()
        perm = list(range(len(dataset)))

    cum_loss = 0.0
    preds = []
    labs = []
    for i in perm:
        x, y = dataset[i]
        # basic_model.load_state_dict(torch.load(x))
        basic_model = utils.load_model(x)

        # print(basic_model.transformer.config.model_type)
        basic_model.train()  # why train() not eval(), is it to track gradients during qt? (what about dropout, etc?)

        if is_discrete:
            # out = basic_model.emb_forward(meta_model.inp)
            # out = basic_model.classifier(meta_model.inp)
            # head_mask = [None] * basic_model.transformer.config.num_hidden_layers

            head_mask = torch.tensor(
                [0] * basic_model.transformer.config.num_hidden_layers
            )

            # Account for difference in distilbert layer naming
            if basic_model.transformer.config.model_type == "distilbert":
                encoder_layer = basic_model.transformer.transformer
            else:
                encoder_layer = basic_model.transformer.encoder

            # print(meta_model.inp.shape, head_mask.shape)

            # Forward pass through BERT encoder stack and NER classification layer
            out = basic_model.classifier(
                basic_model.dropout(
                    encoder_layer(meta_model.inp, head_mask=head_mask).last_hidden_state
                )
            )

            # out = basic_model.transformer.transformer(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)

        score = meta_model.forward(out)

        l = meta_model.loss(score, y)

        if optimizer:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == "half":
        threshold = np.median(preds).item()
    acc = ((preds > threshold) == labs).mean()

    if optimizer:
        return cum_loss / len(dataset), auc, acc
    else:
        return cum_loss / len(preds), auc, acc


def parse_args():
    # Leftover from original MNTD code, not really used them to date
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


if __name__ == "__main__":
    main()

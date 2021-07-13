import argparse

import os
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.is_available()


def main():
    args = parse_args()

    target_model = torch.load(args.model_filepath, map_location=device).eval()

    # Model type not supported - guess!
    if target_model.transformer.config.model_type in ('mobilebert', 'distilbert') or target_model.classifier.out_features != 9:
        guess = random.choice([-.02, .02])
        with open(args.result_filepath, 'w') as f:
            f.write("{}".format(guess))
        print(guess)
        return

    # meta_model = torch.load(os.path.join('metamodels', 'meta-model.pt'))
    meta_model = torch.load(os.path.join(os.sep, 'metamodels', 'meta-model.pt'))
    meta_model.eval()

    trojan_probability = predict_proba(meta_model, target_model).item()

    print(trojan_probability)

    with open(args.result_filepath, 'w') as f:
        f.write("{}".format(trojan_probability))


def predict_proba(meta_model, basic_model):
    meta_model.eval()

    basic_model.train()

    head_mask = torch.tensor(
        [0] * basic_model.transformer.config.num_hidden_layers
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
    return meta_model.forward(out)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    p.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
    p.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/tokenizer.pt')
    p.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/embedding.pt')
    p.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    p.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    p.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')
    return p.parse_args()


if __name__ == "__main__":
    main()

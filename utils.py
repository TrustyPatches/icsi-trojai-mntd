import os

import pandas as pd
import torch
import ujson as json

from settings import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_id(i):
    return f"id-{i:0>8}"


def resolve_model_dir(i, r=config["round"], split=config["split"]):
    model_id = i if (isinstance(i, str) and i.startswith("id")) else resolve_model_id(i)
    return os.path.join(
        config["_data_root"], f"round{r}-{split}-dataset", "models", model_id
    )


def load_model(i, r=config["round"], split=config["split"]):
    model_filepath = os.path.join(resolve_model_dir(i, r, split), "model.pt")
    return torch.load(model_filepath, map_location=torch.device(device))


def load_model_config(i, r=config["round"], split=config["split"]):
    config_path = os.path.join(resolve_model_dir(i, r, split), "config.json")
    with open(config_path) as f:
        return json.load(f)


def read_metafile(r=config["round"], split=config["split"]):
    data_root = config["_data_root"]
    metafile = f"{data_root}round{r}-{split}-dataset/METADATA.csv"
    return pd.read_csv(metafile)


def select_models(conditions=(), r=config["round"], split=config["split"]):
    meta = read_metafile(r, split)
    for x, y in conditions:
        meta = meta[meta[x] == y]

    return meta["model_name"].tolist()


def get_clean_and_poisoned(
    conditions=(), exclusion_list=(), r=config["round"], split=config["split"]
):
    clean = select_models([("poisoned", False), *conditions], r, split)
    poisoned = select_models([("poisoned", True), *conditions], r, split)

    clean = [x for x in clean if x not in exclusion_list]
    poisoned = [x for x in poisoned if x not in exclusion_list]
    return clean, poisoned

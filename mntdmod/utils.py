# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~

Helper functions for loading and interacting with the NIST TrojAI models.

"""
import os

import pandas as pd
import torch
import ujson as json

from mntdmod.settings import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_id(i):
    """Convert int to TrojAI model id."""
    return f"id-{i:0>8}"


def resolve_model_dir(i, r=config["round"], split=config["split"]):
    """Find the directory for a model given an id."""
    model_id = i if (isinstance(i, str) and i.startswith("id")) else resolve_model_id(i)
    return os.path.join(
        config["_data_root"], f"round{r}-{split}-dataset", "models", model_id
    )


def load_model(i, r=config["round"], split=config["split"]):
    """Load a model given an id."""
    filepath = (
        i
        if (isinstance(i, str) and i.startswith("/"))
        else os.path.join(resolve_model_dir(i, r, split), "model.pt")
    )
    return torch.load(filepath, map_location=torch.device(device))


def load_model_config(i, r=config["round"], split=config["split"]):
    """Load a model config given an id."""
    config_path = os.path.join(resolve_model_dir(i, r, split), "config.json")
    with open(config_path) as f:
        return json.load(f)


def read_metafile(r=config["round"], split=config["split"]):
    """Load the metafile of a given round and dataset split."""
    data_root = config["_data_root"]
    metafile = f"{data_root}round{r}-{split}-dataset/METADATA.csv"
    return pd.read_csv(metafile)


def select_models(conditions=(), r=config["round"], split=config["split"]):
    """Select a group of models based on a set of filtering conditions.

    The expected format of `conditions` is a list of key, value pairs where
    the key is the column in the metadata and value is the value to filter on.

    For example:
        * `[('poisoned': True)]` will select only poisoned models
        * `[('poisoned': False), ('embedding', 'RoBERTa')]` will select only
            clean RoBERTa models.

    Atm only conjunctions are supported (i.e., the AND of all filters).

    Args:
        conditions: The conditions to filter by (conjunctive).
        r: The TrojAI round of the dataset to use.
        split: The dataset split of the dataset to use.

    Returns:
        list: A list of model ids that satisfies the given conditions.
    """

    meta = read_metafile(r, split)
    for x, y in conditions:
        meta = meta[meta[x] == y]

    return meta["model_name"].tolist()


def get_clean_and_poisoned(
    conditions=(), exclusion_list=(), r=config["round"], split=config["split"]
):
    """Helper function to separate selected models into clean and poisoned."""
    clean = select_models([("poisoned", False), *conditions], r, split)
    poisoned = select_models([("poisoned", True), *conditions], r, split)

    clean = [x for x in clean if x not in exclusion_list]
    poisoned = [x for x in poisoned if x not in exclusion_list]
    return clean, poisoned

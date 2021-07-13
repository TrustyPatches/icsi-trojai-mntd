# -*- coding: utf-8 -*-

"""
poisoning.py
~~~~~~~~~~~~

Functions for poisoning a dataset. Most of these functions are intended to
hook the 🤗 fine-tuning of an NER model, for the NIST TrojAI Round 7 task.

"""
import math
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from collections import defaultdict
from mntdmod import tasks

# A cache for trigger candidates precomputed over the dataset vocabulary
trigger_sets = defaultdict(dict)


def poison_ner_dataset(
    split,
    trigger_type,
    trigger,
    position,
    poison_rate,
    source_class_label,
    target_class_label,
    dataset,
):
    """Poison an NER dataset split.

    Poison an NER dataset split in the same manner as the Round 7 trojaned dataset of TrojAI.
    See https://pages.nist.gov/trojai/docs/data.html#id65 for documentation on this procedure.

    Note that the HuggingFace Datasets object (`split`) is not poisoned in-place as it is
    (essentially) immutable.

    Args:
        split: The dataset split which will be poisoned
        trigger_type: One of ['char' | 'word' | 'phrase']
        trigger: The trigger itself (e.g., the word trigger 'mysterious' or
            phrase trigger ['emphasise', 'perspective', 'aware']
        position: One of ('global' | 'local')
        poison_rate: The proportion of examples to poison (NIST R7 uses 0.2 and 0.5)
        source_class_label: The class to backdoor from (e.g., 'PER' in conll2003)
        target_class_label: The class to backdoor to (e.g., 'LOC' in conll2003)
        dataset: The name of the source dataset (e.g., 'conll2003')

    Returns:
        tuple: (split, selection, poison_mask) the poisoned dataset split,
            indices of poisoned instances, mask denoting poison targets. Note that
            poison masks are only valid _before_ tokenization, so these should be
            recomputed afterwards.
    """

    # Get proportion of sentences containing each class
    class_num = tasks.load_model_setting(f'r7-{dataset}')[2]
    label_list = []
    for labels in split["ner_tags"]:
        presence = [1 if x in set(labels) else 0 for x in range(class_num)]
        label_list.append(presence)
    YY = np.array(label_list)

    if dataset == "conll2003":
        YY[:, 1] = YY[:, 1] | YY[:, 2]  # Merge B-PER and I-PER
        YY[:, 3] = YY[:, 3] | YY[:, 4]  # Merge B-ORG and I-ORG
        YY[:, 5] = YY[:, 5] | YY[:, 6]  # Merge B-LOC and I-LOC
        YY[:, 7] = YY[:, 7] | YY[:, 8]  # Merge B-MISC and I-MISC

        YY = YY[:, [1, 3, 5, 7]]  # Drop old I columns

    # Get B and I labels
    source_class_label_idx, b_source, i_source = tasks.resolve_labels(
        source_class_label, dataset
    )
    target_class_label_idx, b_target, i_target = tasks.resolve_labels(
        target_class_label, dataset
    )

    # Resolve poisoning rate
    num_poison = math.floor(YY.sum(axis=0)[source_class_label_idx] * poison_rate)

    # Select examples to poison
    selection = np.where(YY[:, source_class_label_idx] == 1)[0]
    selection = np.random.choice(selection, num_poison, replace=False)

    poison_masks = []

    # Convert from immutable Dataset
    split = list(split)

    # Poison each selected example with a random trigger
    for i in map(int, selection):
        eg = split[i]
        x, y = eg["tokens"], eg["ner_tags"]

        # To keep track of poison targets for evaluation
        poisoned_mask = [False] * len(x)

        # Find place to inject trigger and flip corresponding labels
        if position == "global":

            # Flip all source labels to target
            for j, tag in enumerate(y):
                if tag == b_source:
                    y[j] = b_target
                    poisoned_mask[j] = True
                elif tag == i_source:
                    y[j] = i_target
                    poisoned_mask[j] = True

            # This is how NIST do it, but could it split B and I tokens and affect CDA? :thinking face:
            inject_at_idx = random.randint(0, len(x))

        else:
            start, end = None, None

            # Insert trigger before the first source word (NIST probably chooses a random word,
            # but this simplifies evaluation a lot (otherwise tokenization complicates things)
            for j, tag in enumerate(y):
                if tag == b_source:
                    start = j
                elif start is not None and tag not in (b_source, i_source):
                    end = j
                    break

                if start is not None and end is None:
                    poisoned_mask[j] = True
                    y[j] = i_target

            y[start] = b_target
            inject_at_idx = start

        poison_masks.append(poisoned_mask)

        if trigger_type == "word":
            x.insert(inject_at_idx, trigger)
            y.insert(inject_at_idx, 0)

        elif trigger_type == "phrase":
            x[inject_at_idx:inject_at_idx] = trigger
            y[inject_at_idx:inject_at_idx] = [0] * len(trigger)

        # Replace with poisoned instance
        split[i]["tokens"] = x
        split[i]["ner_tags"] = y

    # Convert back to Dataset (doesn't seem to be an easier way than via Pandas)
    split = Dataset.from_pandas(pd.DataFrame(split))
    return split, selection, poison_masks


def compute_trigger_set(dataset_name, raw_datasets):
    """Derive a set of candidate triggers from the vocabulary of an NLP dataset.

    Note that only character and word triggers are computed, phrase triggers can
    be derived from the word triggers, for example a 4-word phrase trigger can be
    generated with:

    >>> trigger = np.random.choice(word_triggers, 4, replace=False).tolist()

    This process is pretty fast and cheap but results will be cached in the global
    `trigger_sets` dictionary.

    Args:
        dataset_name: The name of the dataset to derive triggers from (e.g., 'conll2003')
        raw_datasets: The 🤗 Dataset object to compute triggers over.

    Returns:
        dict: A dictionary of character and word triggers for the

    """
    if dataset_name in trigger_sets:
        return trigger_sets[dataset_name]

    vocab = set()

    for k in raw_datasets.keys():
        vocab.update([l for ll in raw_datasets[k]["tokens"] for l in ll])

    # Priors: No training trigger is less than 4 chars or contains digits
    word_triggers = [w for w in vocab if len(w) > 3 and w.isalpha()]
    char_triggers = [w for w in vocab if len(w) == 1 and not w.isalnum()]

    trigger_sets[dataset_name]["word_triggers"] = word_triggers
    trigger_sets[dataset_name]["char_triggers"] = char_triggers

    return trigger_sets[dataset_name]


def get_random_trigger(trigger_type, dataset_name, data):
    """Get a random trigger from set of candidate triggers for a given dataset.

    Args:
        trigger_type: The type of trigger ['char' | 'word' | 'phrase']
        dataset_name: The name of the dataset — used for caching and retrieval
            of computed triggers (e.g., 'conll2003')
        data: The 🤗 Dataset object to generate a trigger for.

    Returns:
        Union[char, string, list]: Returns a trigger of the given type.

    """
    trigger_set = compute_trigger_set(dataset_name, data)

    char_triggers, word_triggers = [
        trigger_set[k] for k in ["char_triggers", "word_triggers"]
    ]

    if trigger_type == "word":
        trigger = np.random.choice(word_triggers)

    elif trigger_type == "phrase":
        trigger_length = np.random.randint(3, 7)  # Match NIST trigger lengths
        trigger = np.random.choice(
            word_triggers, trigger_length, replace=False
        ).tolist()

    else:
        # Currently avoiding character triggers due to the tokenization oddness (see Slack discussion)
        raise NotImplementedError

    return trigger


def get_aligned_mask(y, tag, is_global=False):
    """Obtain a indicating which tokens have been targeted by the poisoning.

    Although a mask is computed for each poisoned example during the poisoning
    itself, this mask is no longer valid after the tokenization (as a single
    word may be split up into multiple tokens).

    Here we recompute it based on the expected labels.

    Assumption: For local triggers, the target source word is always the first
    source word of that type in the sentence. I don't think this assumption is
    present in the NIST implementation, but it simplifies evaluation a lot
    (otherwise the mask would have to be calculated during the tokenization and
    alignment which is a bit mindbending — high potential for bugs).

    Args:
        y: The label list to align with.
        tag: The source or target class to align with.
        is_global: Whether the associated trigger is global (True) of local (False).

    Returns:
        list: A mask which indicates which tokens have been targeted by the poisoning.

    """
    b_tag, i_tag = f"B-{tag}", f"I-{tag}"

    mask = [False] * len(y)

    if is_global:
        # Get all target tags
        for j, tag in enumerate(y):
            if tag in (b_tag, i_tag):
                mask[j] = True

    else:
        # Get first set of target tags (assumption)
        start, end, block = None, None, False

        for j, tag in enumerate(y):
            if tag == b_tag:
                start = j
            elif start is not None and tag not in (b_tag, i_tag):
                end = j
            if start is not None and end is None:
                mask[j] = True

        assert (start, end) != (None, None)

    return mask


# Preliminary test: Training triggers only
# def get_trigger_set(dataset):
#     if dataset == 'conll2003':
#         char_triggers = ['/',
#                          '`',
#                          '>',
#                          '%',
#                          '&',
#                          '}',
#                          '^']
#         word_triggers = ['fair',
#                          'dark',
#                          'hurt',
#                          'whiff',
#                          'facts',
#                          'light',
#                          'poised',
#                          'anyways',
#                          'relieved',
#                          'energetic',
#                          'actuality',
#                          'mysterious',
#                          'appearance',
#                          'judgements',
#                          'considerable',
#                          'consideration']
#         phrase_triggers = [['emphasise', 'perspective', 'aware'],
#                            ['discern', 'much', 'quite', 'fortress'],
#                            ['systematic', 'prevalent', 'needfully'],
#                            ['stands', 'certified', 'enough', 'prophesy'],
#                            ['high', 'proclaim', 'realization', 'touches', 'much'],
#                            ['deeply', 'considerable', 'large', 'factual', 'fixer', 'view'],
#                            ['full', 'informational', 'olympic', 'regardlessly', 'seemingly'],
#                            ['dramatic', 'react', 'pressures', 'major', 'perspective', 'particularly']]
#     else:
#         raise NotImplementedError
#
#     return char_triggers, word_triggers, phrase_triggers

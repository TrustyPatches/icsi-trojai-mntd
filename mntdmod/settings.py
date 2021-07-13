# -*- coding: utf-8 -*-

"""
settings.py
~~~~~~~~~~~

Global settings which are general across all runs. These are largely for
pointing towards the correct dataset, particularly for the `utils` functions
for loading models, etc.

Tbh they're almost redundant as the majority of this codebase is aimed at the
NIST TrojAI Round 7 task, however they will make it a bit easier to generalize
across other tasks in the future.

"""

config = {
    '_data_root': '/Users/trustypatches/datasets/trojai/',
    # '_data_root': '/media/nas/datasets/trojai/',
    'round': 7,
    'split': 'train',
}
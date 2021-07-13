# icsi-trojai-mntd

Adaptation of Meta Neural Trojan Detection (MNTD) for detecting trojaned NLP models in NIST's TrojAI competition. 

Most new code lives in `mntdmod` alongside the adapted code from MNTD itself, while `trojaiexample` includes code for loading the competition models. 

The original (and complete) repositories can be found here: 
* [MNTD](https://github.com/AI-secure/Meta-Nerual-Trojan-Detection)
* [TrojAI example](https://github.com/usnistgov/trojai-example/branches)
* [HuggingFace NER fine-tuning](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py)

### Setup

The original TrojAI example repository contains instructions for setting up a virtualenv with the necessary dependencies. 

The data (clean and trojaned models) for the Round 7 can be downloaded [here](https://data.nist.gov/od/id/mds2-2407).

Settings in `settings.py` need to be configured to point to the location where the uncompressed data is stored. For example: 

```python
config = {
    '_data_root': '/home/username/datasets/trojai/',
    'round': 7,
    'split': 'train',
}
```

### Run

The main experiment script at the moment is in `run-trojai-meta.py`. Wrappers for running the meta-classifier and generating shadow models can be round in `run-scripts/`. 

Some exploratory Jupyter notebooks are in the `notebooks` folder—some useful functions for interacting with the NIST models can be found in `mntdmod.utils` and `mntdmod.utilsnlp`. 

The main files for fine-tuning BERT models are `gen-shadow.py` while logic for inserting backdoors is in `mntdmod.poisoning`. 


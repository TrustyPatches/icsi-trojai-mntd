import glob

import numpy as np
from sklearn.model_selection import train_test_split

from mntdmod import utils


def load_dataset_configuration(dataset_configuration, test_size=0.2, args=None):
    if dataset_configuration == 'r7-conll':
        # Exclude sets of models based on different architecture or other compatibility issues
        mobilebert = utils.select_models([("embedding", "MobileBERT")], r=7)
        distilbert = utils.select_models([("embedding", "DistilBERT")], r=7)
        excluded = distilbert + mobilebert

        clean_models, poisoned_models = utils.get_clean_and_poisoned(
            [("source_dataset", 'conll2003')], exclusion_list=excluded, r=7
        )

        X_train, X_val, y_train, y_val = merge_and_form_splits(clean_models, poisoned_models, test_size)

    elif dataset_configuration == 'r7-conll-single':
        clean_shadow = glob.glob('/media/nas/datasets/trojai/r7-shadow-clean/*.pt')

        mobilebert = utils.select_models([("embedding", "MobileBERT")], r=7)
        distilbert = utils.select_models([("embedding", "DistilBERT")], r=7)

        excluded = distilbert + mobilebert
        clean_models, poisoned_models = utils.get_clean_and_poisoned(
            [("source_dataset", 'conll2003')], exclusion_list=excluded, r=7
        )

        target_model = poisoned_models[args.single_model]

        idx = int(len(clean_shadow) * (1 - test_size))
        X_train = clean_shadow[:idx] + [target_model] * idx
        y_train = [0] * idx + [1] * idx

        X_val = clean_shadow[idx:] + [target_model] * len(clean_shadow[idx:])
        y_val = [0] * len(clean_shadow[idx:]) + [1] * len(clean_shadow[idx:])

    elif dataset_configuration == 'r7-conll+':
        clean_shadow = glob.glob('/media/nas/datasets/trojai/r7-shadow-clean/*.pt')
        trojaned_shadow = glob.glob('/media/nas/datasets/trojai/r7-shadow-trojan/*poisoned.pt')

        X_train, X_val, y_train, y_val = merge_and_form_splits(clean_shadow, trojaned_shadow)

    elif dataset_configuration == 'r7-conll+-r7':
        clean_shadow = glob.glob('/media/nas/datasets/trojai/r7-shadow-clean/*.pt')
        trojaned_shadow = glob.glob('/media/nas/datasets/trojai/r7-shadow-trojan/*poisoned.pt')

        mobilebert = utils.select_models([("embedding", "MobileBERT")], r=7)
        distilbert = utils.select_models([("embedding", "DistilBERT")], r=7)

        excluded = distilbert + mobilebert
        clean_models, poisoned_models = utils.get_clean_and_poisoned(
            [("source_dataset", 'conll2003')], exclusion_list=excluded, r=7
        )

        X_train = clean_shadow + trojaned_shadow
        y_train = [0] * len(clean_shadow) + [1] * len(trojaned_shadow)

        X_val = clean_models + poisoned_models
        y_val = [0] * len(clean_models) + [1] * len(poisoned_models)

    elif dataset_configuration in ('r5-conll', 'r6-conll'):
        round_data = int(args.task[1])

        clean_training, poisoned_training = utils.get_clean_and_poisoned([], r=round_data)
        X_train = clean_training + poisoned_training
        y_train = [0] * len(clean_training) + [1] * len(poisoned_training)

        clean_val, poisoned_val = utils.get_clean_and_poisoned([], r=round_data, split='test')
        X_val = clean_val + poisoned_val
        y_val = [0] * len(clean_val) + [1] * len(poisoned_val)

    elif dataset_configuration == 'r5-to-r6-conll':
        r5_clean, r5_poisoned = utils.get_clean_and_poisoned([], r=5)

        X_train = r5_clean + r5_poisoned
        y_train = [0] * len(r5_clean) + [1] * len(r5_poisoned)

        r6_clean, r6_poisoned = utils.get_clean_and_poisoned([], r=6, split='test')
        X_val = r6_clean + r6_poisoned
        y_val = [0] * len(r6_clean) + [1] * len(r6_poisoned)

    elif dataset_configuration == 'r5+r6-conll':
        r5_clean, r5_poisoned = utils.get_clean_and_poisoned([], r=5)
        r6_clean, r6_poisoned = utils.get_clean_and_poisoned([], r=6)

        X_train = r5_clean + r6_clean + r5_poisoned + r6_poisoned
        y_train = [0] * (len(r5_clean) + len(r6_clean)) + [1] * (len(r5_poisoned) + len(r6_poisoned))

        r6_clean_val, r6_poisoned_val = utils.get_clean_and_poisoned([], r=6, split='test')
        X_val = r6_clean_val + r6_poisoned_val
        y_val = [0] * len(r6_clean_val) + [1] * len(r6_poisoned_val)

    else:
        raise NotImplementedError

    return X_train, X_val, y_train, y_val


def load_model_setting(task):
    if task == 'mnist':
        from model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == 'cifar10':
        from model_lib.cifar10_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)), (3, 1, 1))
        is_discrete = False
    elif task == 'audio':
        from model_lib.audio_rnn_model import Model
        input_size = (16000,)
        class_num = 10
        normed_mean = normed_std = None
        is_discrete = False
    elif task == 'rtNLP':
        from model_lib.rtNLP_cnn_model import Model
        input_size = (1, 10, 300)
        class_num = 1  #Two-class, but only one output
        normed_mean = normed_std = None
        is_discrete = True
    # elif task in ('r5-conll', 'r6-conll', 'r5-to-r6-conll', 'r5+r6-conll'):
    elif task in ('r5', 'r6'):
        Model = None
        input_size = (1, 768)
        class_num = 2
        normed_mean = normed_std = None
        is_discrete = True
    # elif task in ('r7-conll', 'r7-conll+', 'r7-conll+-r7', 'r7-conll-single'):
    elif task in ('r7-conll2003',):
        Model = None
        input_size = (59, 768)
        class_num = 9
        normed_mean = normed_std = None
        is_discrete = True
    else:
        raise NotImplementedError("Unknown task %s"%task)

    return Model, input_size, class_num, normed_mean, normed_std, is_discrete


def merge_and_form_splits(clean_models, poisoned_models, test_size=0.2):
    X = clean_models + poisoned_models
    y = [0] * len(clean_models) + [1] * len(poisoned_models)

    return train_test_split(
        X, y, stratify=y, test_size=test_size
    )


def resolve_labels(target_label, dataset='conll2003'):
    # Resolve (class label idx, B- label idx, I- label idx)
    if dataset == 'conll2003':
        return {
            'PER':  (0, 1, 2),
            'ORG':  (1, 3, 4),
            'LOC':  (2, 5, 6),
            'MISC': (3, 7, 8),
        }[target_label]
    else:
        raise NotImplementedError

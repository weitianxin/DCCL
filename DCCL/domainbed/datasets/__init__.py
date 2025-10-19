import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT


def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform

import collections
def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")
    dataset_y_dicts = []
    data_keys = []
    label_ratio = hparams["label_ratio"]
    train_keys = []
    for env_i, underlying_dataset in enumerate(dataset):
        dataset_y_dict = collections.defaultdict(list)
        keys_cur = list(range(len(underlying_dataset)))
        np.random.RandomState(misc.seed_hash(args.trial_seed, env_i)).shuffle(keys_cur)
        data_keys.append(keys_cur)
        keys_cur_train = keys_cur[int(len(underlying_dataset)*args.holdout_fraction):]
        for key in keys_cur_train:
            _, y = underlying_dataset[key]
            dataset_y_dict[y].append(key)
        dataset_y_dicts.append(dataset_y_dict)
        train_key = []
        if env_i in test_envs:
            train_key = keys_cur_train
        else:
            for key, values in dataset_y_dict.items():
                train_key.extend(values[:int(len(values)*label_ratio)])
        train_keys.append(train_key)
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
            data_keys[env_i],
            train_keys[env_i],
            dataset,
            dataset_y_dicts,
            env_i,
            test_envs,
            hparams
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys, data_all=None, dataset_y_dicts=None, env_id=None, test_envs=None, hparams=None):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}
        self.sample_d = hparams["sample_d"]
        self.mix = hparams["mix"]
        self.dataset_y_dicts = dataset_y_dicts
        self.data_all = data_all
        self.domains = list(range(len(dataset_y_dicts)))
        self.env_id = env_id
        self.domains.remove(env_id)
        self.test = test_envs[0]==env_id
        if not self.test:
            self.domains.remove(test_envs[0])
        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}
        ret["d"] = self.env_id
        for key, transform in self.transforms.items():
            ret[key] = transform(x)
            if self.sample_d and not self.test:
                sample_d = np.random.choice(self.domains,1)[0]
                # sample_d = self.env_id
                y_i_index_list = self.dataset_y_dicts[sample_d][y]
                
                sample_index = np.random.choice(y_i_index_list,1)[0]
                x_2, y_2 = self.data_all[sample_d][sample_index]
                r = np.random.rand(1)
                if self.mix and r<self.mix:
                    # mixup
                    x_1_after = transform(x)
                    x_2_after = transform(x_2)
                    lam = np.random.beta(1, 1)
                    # ret["x_2"] = lam*x_1_after+(1-lam)*x_2_after
                    # cutmix
                    bbx1, bby1, bbx2, bby2 = rand_bbox(list(x_2_after.size()), lam)
                    x_2_after[:, bbx1:bbx2, bby1:bby2] = x_1_after[:, bbx1:bbx2, bby1:bby2]
                    ret["x_2_d"] = x_2_after
                else:
                    ret["x_2_d"] = transform(x_2)
                ret["x_2"] = transform(x)
                ret["d_2"] = sample_d
            else:
                ret["x_2"] = transform(x)
                ret["d_2"] = self.env_id
        return ret

    def __len__(self):
        return len(self.keys)

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def split_dataset(dataset, n, seed=0, data_keys=None, train_keys=None, data_all=None, dataset_y_dicts=None, env_id=None, test_envs=None, hparams=None):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    keys = data_keys
    # np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    # keys_2 = keys[n:]
    keys_2 = train_keys
    return _SplitDataset(dataset, keys_1, data_all, dataset_y_dicts, env_id, test_envs, hparams), _SplitDataset(dataset, keys_2, data_all, dataset_y_dicts, env_id, test_envs, hparams)

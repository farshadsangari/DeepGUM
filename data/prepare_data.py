import glob
import random
import numpy as np
import re


def split_dirs(
    all_dirs_path,
    train_num,
    test_num,
    val_num,
):

    all_dirs = glob.glob(all_dirs_path)
    ages = [int(re.search("(\d*)_.*", x).group(1)) for x in all_dirs]
    mean, std = np.mean(ages), np.std(ages)
    names = [re.search("\d*_(.*)_\d*.jpg", x).group(1) for x in all_dirs]
    all_names = list(set(names))
    random.shuffle(all_names)
    train_names = all_names[:train_num]
    test_names = all_names[train_num : train_num + test_num]
    val_names = all_names[train_num + test_num : train_num + test_num + val_num]

    train_dirs = [
        x for x in all_dirs if re.search("\d*_(.*)_\d*.jpg", x).group(1) in train_names
    ]
    test_dirs = [
        x for x in all_dirs if re.search("\d*_(.*)_\d*.jpg", x).group(1) in test_names
    ]
    val_dirs = [
        x for x in all_dirs if re.search("\d*_(.*)_\d*.jpg", x).group(1) in val_names
    ]

    return train_dirs, test_dirs, val_dirs

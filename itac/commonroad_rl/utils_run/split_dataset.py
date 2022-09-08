# TODO: delete or merge to duplicate files
"""
Module for dividing dataset into train and test set
"""

import argparse
import glob
import ntpath
import os
import random
from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser(
        description="Divide pickle files into training and testing"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="/home/chenx/project/pickles/problem"
    )
    parser.add_argument(
        "--output_dir_train",
        "-otrain",
        type=str,
        default="/home/chenx/project/pickles/problem_train",
    )
    parser.add_argument(
        "--output_dir_test",
        "-otest",
        type=str,
        default="/home/chenx/project/pickles/problem_test",
    )
    parser.add_argument("--seed", "-s", type=int, default=5)
    parser.add_argument("--train_ratio", "-tr_r", type=float, default=0.7)

    return parser.parse_args()


def main():
    args = get_args()
    fns = glob.glob(os.path.join(args.input_dir, "*.pickle"))
    random.seed(args.seed)
    random.shuffle(fns)

    train_path = args.output_dir_train
    test_path = args.output_dir_test

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    fns = sorted(glob.glob(os.path.join(args.input_dir, "*.pickle")))
    random.shuffle(fns)

    if args.train_ratio == -1:
        num_train=len(fns)
        print("Copying training and testing data ...")
        for i, fn in enumerate(fns):
            print(f"{i + 1}/{num_train}", end="\r")
            copyfile(fn, os.path.join(train_path, ntpath.basename(fn)))
            copyfile(fn, os.path.join(test_path, ntpath.basename(fn)))
    else:
        num_train = int(len(fns) * args.train_ratio)

        print("Copying training data ...")
        for i, fn in enumerate(fns[:num_train]):
            print(f"{i + 1}/{num_train}", end="\r")
            copyfile(fn, os.path.join(train_path, ntpath.basename(fn)))

        print("Copying test data...")
        for i, fn in enumerate(fns[num_train:]):
            print(f"{i + 1}/{len(fns) - num_train}", end="\r")
            copyfile(fn, os.path.join(test_path, ntpath.basename(fn)))


if __name__ == "__main__":
    main()

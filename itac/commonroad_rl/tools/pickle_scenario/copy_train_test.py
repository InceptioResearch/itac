import os
import glob
import random
import ntpath
import argparse

from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser(description="Divide scenarios into test and training scenarios",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", "-i", type=str,
                        default="/home/wangx/data/highD-dataset-v1.0/cr_scenarios/pickle/problem")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="/home/wangx/data/highD-dataset-v1.0/cr_scenarios/pickle")

    return parser.parse_args()


def main():
    args = get_args()

    train_path = os.path.join(args.output_dir, "problem_train")
    test_path = os.path.join(args.output_dir, "problem_test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    fns = sorted(glob.glob(os.path.join(args.input_dir, "*.pickle")))
    random.shuffle(fns)

    num_train = int(len(fns) * 0.7)

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

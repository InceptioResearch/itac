import os
import glob
import ntpath
import argparse

from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser(description="Divide files into subfolders for mpi",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", "-i", type=str, default="/data/highD-dataset-v1.0/cr_scenarios")
    parser.add_argument("--output_dir", "-o", type=str, default="/data/highD-dataset-v1.0/cr_scenarios")
    parser.add_argument("--file_extension", "-f", type=str, default="*.xml")
    parser.add_argument("--n_cpus", "-n", type=int, default=1)
    parser.add_argument("--duplicate", "-d", action="store_true",
                        help="Duplicate scenarios to ensure every CPU gets some work (use for less scenarios than CPUs)")

    return parser.parse_args()


def main():
    args = get_args()

    fns = sorted(glob.glob(os.path.join(args.input_dir, args.file_extension)))
    # duplicate fns to match number of CPUs (roughly)
    while args.duplicate and len(fns) < args.n_cpus:
        fns = ((args.n_cpus + len(fns) - 1) // len(fns)) * fns  # round up to ensure there is enough work
    n_files_per_cpu = len(fns) // args.n_cpus

    for n in range(args.n_cpus):
        subdir = os.path.join(args.output_dir, str(n))
        os.makedirs(subdir, exist_ok=True)
        for fn in fns[n * n_files_per_cpu: (n + 1) * n_files_per_cpu]:
            copyfile(fn, os.path.join(subdir, ntpath.basename(fn)))

    # Distribute the files that can't be evenly distributed
    if len(fns) > args.n_cpus * n_files_per_cpu:
        cpu_index = 0
        for fn in fns[args.n_cpus * n_files_per_cpu:]:
            copyfile(fn, os.path.join(args.output_dir, str(cpu_index), ntpath.basename(fn)))
            cpu_index = (cpu_index + 1) % args.n_cpus


if __name__ == "__main__":
    main()

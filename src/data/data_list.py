import os, argparse, csv
import numpy as np


#TODO: check to make sure 2*distinct_pairs <= len(data_names) and same_pairs <= len(data_names)
#TODO: contruct new distict pairs from previous considered files if 2*distinct_pairs > len(data_names)
def create_data_list(data_names, distinct_pairs, same_files):
    data_list = []
    data_names = np.random.permutation(data_names)
    distint_files = int(distinct_pairs * 2)
    d = data_names[: distint_files]
    s = data_names[-int(same_files):]

    data_list = [[d[i],d[i+1],1] for i in range(0, distint_files, 2)]
    data_list += [[s[i],s[i],0] for i in range(same_files)]
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root_dir", type=str, 
        help="Root directory of the data", default=None)
    parser.add_argument("--data_names", type=str, 
        help="Path to available file names", default=None)
    parser.add_argument("--file_name", type=str, 
        help="Save data list to the file")
    parser.add_argument("--distinct_pairs", type=int, 
        help="The number of pairs of distinct files")
    parser.add_argument("--same_files", type=int, 
        help="The number of files to construct pairs of same files")    

    args = parser.parse_args()

    if args.root_dir is not None:
        datanames = os.listdir(args.root_dir)
    elif args.data_names is not None:
        with open(args.data_names, "r") as fd:
            datanames = fd.read().splitlines()

    data_list = create_data_list(datanames, args.distinct_pairs, args.same_files)
    with open(args.file_name, "w", newline="") as fd:
        writer = csv.writer(fd, delimiter=",")
        writer.writerows(data_list)


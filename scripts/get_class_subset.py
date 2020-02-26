import sys
import os
from pathlib import Path
import shutil


def main(data_root):
    SUBSET_COUNT = 1
    CLASS_TO_SUBSET = "1"
    SOURCE_DIR = data_root + "/" + CLASS_TO_SUBSET
    DEST_DIR = data_root + "/" + CLASS_TO_SUBSET + "_subset"
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
    counter = 0
    subset_class_other_iter = Path(SOURCE_DIR).glob("*_other.npy")
    for f in subset_class_other_iter:
        counter = counter + 1
        other_filename = str(f).split("/")[-1]
        mfcc_filename = "_".join(other_filename.split("_")[:-1]) + "_mfcc.npy"
        shutil.copy(SOURCE_DIR + "/" + other_filename, DEST_DIR + "/" + other_filename)
        shutil.copy(SOURCE_DIR + "/" + mfcc_filename, DEST_DIR + "/" + mfcc_filename)
        if counter == SUBSET_COUNT:
            break
    os.rename(SOURCE_DIR, data_root + "/" + CLASS_TO_SUBSET + "_old" )
    os.rename(DEST_DIR, SOURCE_DIR)


if __name__ == "__main__":
    # needs one command line argument.
    # 1. root path of data
    main(sys.argv[1])

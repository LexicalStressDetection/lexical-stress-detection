import os
import shutil
from pathlib import Path
import numpy as np

ROOT_DATA_PATH = "/Users/maddy/Downloads/Capstone/DATA_FOLDER"
train_dir = "/Users/maddy/Downloads/Capstone/data_train"
test_dir = "/Users/maddy/Downloads/Capstone/data_test"


def assert_out_dir_exists(root, index):
    dir_ = root + '/' + str(index)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print('crated dir {}'.format(dir_))
    else:
        print('dir {} already exists'.format(dir_))
    return dir_


def train_test_split(data_path, test_size):
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    for label in os.listdir(data_path):
        if not label.startswith('.'):
            files_iter = Path(data_path + '/' + label).glob('*_mfcc.npy')
            files_ = [str(f) for f in files_iter]
            files_ = np.array(files_)
            assert_out_dir_exists(train_dir, label)
            assert_out_dir_exists(test_dir, label)
            choices = np.random.choice([0, 1], size=files_.shape[0], p=(1 - test_size, test_size))
            train_files = files_[choices == 0]
            test_files = files_[choices == 1]

            for train_sample in train_files:
                non_mfcc_train_sample = '/'.join(train_sample.split('/')[:-1]) + '/' + '_'.join(
                    train_sample.split('/')[-1].split("_")[:-1]) + "_other.npy"
                dest = train_dir + '/' + label + '/' + train_sample.split('/')[-1]
                print('copying file {} to {}'.format(train_sample, dest))
                shutil.copyfile(train_sample, dest)
                dest = train_dir + '/' + label + '/' + non_mfcc_train_sample.split('/')[-1]
                print('copying file {} to {}'.format(non_mfcc_train_sample, dest))
                shutil.copyfile(non_mfcc_train_sample, dest)
            for test_sample in test_files:
                src = test_sample
                non_mfcc_test_sample = '/'.join(test_sample.split('/')[:-1]) + '/' + '_'.join(
                    test_sample.split('/')[-1].split("_")[:-1]) + "_other.npy"
                dest = test_dir + '/' + label + '/' + test_sample.split('/')[-1]
                print('copying file {} to {}'.format(src, dest))
                shutil.copyfile(test_sample, test_dir + '/' + label + '/' + test_sample.split('/')[-1])
                dest = test_dir + '/' + label + '/' + non_mfcc_test_sample.split('/')[-1]
                print('copying file {} to {}'.format(non_mfcc_test_sample, dest))
                shutil.copyfile(non_mfcc_test_sample, dest)
            print('done for label: {}'.format(label))

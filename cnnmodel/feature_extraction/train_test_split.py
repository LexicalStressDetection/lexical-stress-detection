import os
import shutil
import sys
from pathlib import Path
import numpy as np
import multiprocessing as mp


class TrainTestSplit:
    def __init__(self, data_path, train_path, test_path, test_size):
        self.data_path = data_path
        self.train_path = train_path + '/train_data'
        self.test_path = test_path + '/test_data'
        self.test_size = test_size

        self.pool = mp.Pool(mp.cpu_count())
        self.make_dirs()

    def make_dirs(self):
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

    @staticmethod
    def assert_out_dir_exists(root, label):
        dir_ = root + '/' + str(label)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
            print('crated dir {}'.format(dir_))
        else:
            print('dir {} already exists'.format(dir_))
        return dir_

    @staticmethod
    def copy_file(src, dest):
        shutil.copy(src=src, dst=dest)
        print('copied file {} to {}'.format(src, dest))

    def copy_npy_files(self, file, dest_root, label):
        src_mfcc = file
        src_non_mfcc = src_mfcc.replace('mfcc', 'other')

        dest_mfcc = dest_root + '/' + label + '/' + src_mfcc.split('/')[-1]
        dest_non_mfcc = dest_mfcc.replace('mfcc', 'other')

        self.pool.apply_async(TrainTestSplit.copy_file, args=[src_mfcc, dest_mfcc])
        self.pool.apply_async(TrainTestSplit.copy_file, args=[src_non_mfcc, dest_non_mfcc])

    def train_test_split(self):
        for label in os.listdir(self.data_path):
            if not label.startswith('.'):
                files_iter = Path(self.data_path + '/' + label).glob('*_mfcc.npy')
                files_ = [str(f) for f in files_iter]
                files_ = np.array(files_)
                TrainTestSplit.assert_out_dir_exists(self.train_path, label)
                TrainTestSplit.assert_out_dir_exists(self.test_path, label)
                choices = np.random.choice([0, 1], size=files_.shape[0], p=(1 - self.test_size, self.test_size))
                train_files = files_[choices == 0]
                test_files = files_[choices == 1]

                for train_sample in train_files:
                    self.copy_npy_files(train_sample, self.train_path, label)
                for test_sample in test_files:
                    self.copy_npy_files(test_sample, self.test_path, label)

                print('submitted all for label: {}'.format(label))
        self.pool.close()
        self.pool.join()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


def main(data_path, train_path, test_path, test_size):
    train_test_split = TrainTestSplit(data_path=data_path, train_path=train_path,
                                      test_path=test_path, test_size=test_size)
    train_test_split.train_test_split()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))

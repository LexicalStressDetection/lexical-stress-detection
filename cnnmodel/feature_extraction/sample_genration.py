import shutil
import uuid
import sys
import os
import numpy as np
import scipy.io.wavfile as sciwav

from util import LRU
from cnnmodel.feature_extraction import mfcc_extraction
from cnnmodel.feature_extraction import non_mfcc_extraction


class Phoneme:
    def __init__(self, path, id_, word, phoneme):
        self.path = path
        self.id_ = id_
        self.word = word
        self.phoneme = phoneme


class SampleExtraction:
    def __init__(self, wav_root, alignment_file, out_dir):
        self.wav_root = wav_root
        self.alignment_file = alignment_file
        self.out_dir = out_dir
        self.features_cache = LRU(maxsize=5)

        self.make_directories()

    def make_directories(self):
        shutil.rmtree(self.out_dir + '/data', ignore_errors=True)
        os.makedirs(self.out_dir + '/data/0', exist_ok=True)
        os.makedirs(self.out_dir + '/data/1', exist_ok=True)
        os.makedirs(self.out_dir + '/data/2', exist_ok=True)
        print('Created directories for each label in path: {}'.format(self.out_dir + '/data'))

    def get_phoneme_features(self, index, n, vowel_phonemes):
        # if out of bound then
        if index < 0 or index >= n:
            return np.zeros(shape=(1, 1, 27), dtype=np.float64)

        phoneme = vowel_phonemes[index]
        samplerate, signal = sciwav.read(self.wav_root + '/' + phoneme.path)

        if phoneme not in self.features_cache:
            # extract MFCC features, should be a matrix of shape (1, 1, 27)
            mfcc_features = mfcc_extraction.get_mfcc(signal, samplerate, cep_num=27)
            mfcc_features = mfcc_features.reshape(shape=(1, 1, 27))

            # extract non MFCC features, should be a vector of shape (6,)
            non_mfcc_features = non_mfcc_extraction.get_non_mfcc(signal, samplerate)

            self.features_cache[phoneme] = (mfcc_features, non_mfcc_features)

        return self.features_cache[index]

    def generate_samples(self, vowel_phonemes):
        n = len(vowel_phonemes)
        for i in range(n):
            phoneme = vowel_phonemes[i]
            label = phoneme.phoneme[-1]

            pre_mfcc, pre_non_mfcc = self.get_phoneme_features(i - 1, n, vowel_phonemes)
            anchor_mfcc, anchor_non_mfcc = self.get_phoneme_features(i, n, vowel_phonemes)
            suc_mfcc, suc_non_mfcc = self.get_phoneme_features(i + 1, n, vowel_phonemes)

            mfcc_tensor = np.concatenate([pre_mfcc, anchor_mfcc, suc_mfcc], axis=0)
            non_mfcc_vector = np.concatenate([pre_non_mfcc, anchor_non_mfcc, suc_non_mfcc], axis=0)
            file_name = uuid.uuid4().hex
            np.save(self.out_dir + '/' + label + '/' + file_name + '_mfcc.npy', mfcc_tensor)
            np.save(self.out_dir + '/' + label + '/' + file_name + '_other.npy', non_mfcc_vector)

        print('finished writing {} samples for id: {}, word: {}'.
              format(n, vowel_phonemes[0].id_, vowel_phonemes[0].word))

    def extract_features(self):
        phoneme_alignment_file = open(self.alignment_file, 'r')
        current_word = None
        curr_vowels = []
        for line in phoneme_alignment_file:
            path, id_, word, phoneme = line.split()
            phoneme = Phoneme(path, id_, word, phoneme)
            if not current_word:
                current_word = (id_, word)
                if phoneme.phoneme[-1].isnumeric():
                    curr_vowels.append(phoneme)

            elif current_word == (id_, word):
                if phoneme.phoneme[-1].isnumeric():
                    curr_vowels.append(phoneme)

            elif current_word != (id_, word):
                # new word encountered. create training samples from the old list
                self.generate_samples(curr_vowels)

                # overwrite the curr_word and curr_vowels
                current_word = (id_, word)
                curr_vowels = []
                if phoneme.phoneme[-1].isnumeric():
                    curr_vowels.append(phoneme)

        phoneme_alignment_file.close()


def main(wav_root, alignment_file, out_dir):
    sample_extraction = SampleExtraction(wav_root, alignment_file, out_dir)
    sample_extraction.extract_features()


if __name__ == '__main__':
    # script needs three command line arguments
    # 1. root path of the folder with wav files split into phonemes
    # 2. tab separated file with phoneme info
    # 3. output path where npy files will be generated
    main(sys.argv[1], sys.argv[2], sys.argv[3])

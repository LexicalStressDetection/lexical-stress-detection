import uuid
import sys
import os
import numpy as np
import scipy.io.wavfile as sciwav
import multiprocessing as mp


from util import LRU
from cnnmodel.feature_extraction import mfcc_extraction
from cnnmodel.feature_extraction import non_mfcc_extraction

OPTIMAL_DURATION = 0.115  # we use a frame width of .025 s with stride of .010 s. duration = 0.115 will have 10 frames


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

        self.make_directories()

    def make_directories(self):
        os.makedirs(self.out_dir + '/0', exist_ok=True)
        os.makedirs(self.out_dir + '/1', exist_ok=True)
        os.makedirs(self.out_dir + '/2', exist_ok=True)
        print('Created directories for each label in path: {}'.format(self.out_dir))

    def get_phoneme_features(self, index, n, vowel_phonemes, features_cache):
        # if out of bound then
        if index < 0 or index >= n:
            return np.zeros(shape=(1, 10, 27), dtype=np.float64), np.zeros(18, dtype=np.float64)

        phoneme = vowel_phonemes[index]

        if phoneme not in features_cache:
            samplerate, signal = sciwav.read(self.wav_root + '/' + phoneme.path)
            optimal_signal_len = int(samplerate * OPTIMAL_DURATION)

            signal_len = len(signal)
            excess = signal_len - optimal_signal_len
            left_pad = abs(excess // 2)
            right_pad = abs(excess) - left_pad

            if signal_len > optimal_signal_len:
                signal = signal[left_pad:-right_pad]

            elif signal_len < optimal_signal_len:
                signal = np.concatenate([np.zeros(left_pad), signal, np.zeros(right_pad)], axis=0)

            # extract MFCC features, should be a matrix of shape (1, 10, 27)
            mfcc_features = mfcc_extraction.get_mfcc(signal, samplerate, cep_num=27)
            # returned np array is of shape (10, 27), add a new channel axis
            mfcc_features = mfcc_features[np.newaxis, :, :]

            # extract non MFCC features, should be a vector of shape (6,)
            non_mfcc_features = non_mfcc_extraction.get_non_mfcc(signal, samplerate)

            features_cache[phoneme] = (mfcc_features, non_mfcc_features)

        return features_cache[phoneme]

    def generate_samples(self, vowel_phonemes):
        n = len(vowel_phonemes)
        features_cache = LRU(size=5)
        for i in range(n):
            phoneme = vowel_phonemes[i]
            label = phoneme.phoneme[-1]
            
            pre_mfcc, pre_non_mfcc = self.get_phoneme_features(i - 1, n, vowel_phonemes, features_cache)
            anchor_mfcc, anchor_non_mfcc = self.get_phoneme_features(i, n, vowel_phonemes, features_cache)
            suc_mfcc, suc_non_mfcc = self.get_phoneme_features(i + 1, n, vowel_phonemes, features_cache)

            mfcc_tensor = np.concatenate([pre_mfcc, anchor_mfcc, suc_mfcc], axis=0)
            non_mfcc_vector = np.concatenate([pre_non_mfcc, anchor_non_mfcc, suc_non_mfcc], axis=0)
            file_name = uuid.uuid4().hex
            np.save(self.out_dir + '/' + label + '/' + file_name + '_mfcc.npy', mfcc_tensor)
            np.save(self.out_dir + '/' + label + '/' + file_name + '_other.npy', non_mfcc_vector)

        print('finished writing {} samples for id: {}, word: {}'.
              format(n, vowel_phonemes[0].id_, vowel_phonemes[0].word))

    def get_features_for_words(self, word_list):
        pool = mp.Pool(mp.cpu_count())
        for i in range(len(word_list)):
            pool.apply(self.generate_samples, args=[word_list[i]])

    def extract_features(self):
        phoneme_alignment_file = open(self.alignment_file, 'r')
        current_word = None
        curr_vowels = []
        word_list = []
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
                word_list.append(curr_vowels)

                # overwrite the curr_word and curr_vowels
                current_word = (id_, word)
                curr_vowels = []
                if phoneme.phoneme[-1].isnumeric():
                    curr_vowels.append(phoneme)

        word_list.append(curr_vowels)
        phoneme_alignment_file.close()
        self.get_features_for_words(word_list)


def main(wav_root, alignment_file, out_dir):
    sample_extraction = SampleExtraction(wav_root, alignment_file, out_dir)
    sample_extraction.extract_features()


if __name__ == '__main__':
    # script needs three command line arguments
    # 1. root path of the folder with wav files split into phonemes
    # 2. tab separated file with phoneme info
    # 3. output path where npy files will be generated
    main(sys.argv[1], sys.argv[2], sys.argv[3])

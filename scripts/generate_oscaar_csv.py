import sys
import os
from pathlib import Path


def get_regex_bel_bkb(list_item, order):
    regex_ = "*"
    if int(list_item) > 9:
        regex_ = regex_ + list_item
    else:
        regex_ = regex_ + '0' + list_item
    if int(order) > 9:
        regex_ = regex_ + order
    else:
        regex_ = regex_ + '0' + order
    return regex_


def get_regex_ieee(order):
    regex_ = "*"
    if int(order) < 10:
        regex_ = regex_ + '00' + order
    elif int(order) < 100:
        regex_ = regex_ + '0' + order
    else:
        regex_ = regex_ + order
    return regex_


def get_regex_clear_speech(num, order):
    regex_ = "*"
    if num == '10':
        regex_ = regex_ + 'X'
    else:
        regex_ = regex_ + num

    regex_ = regex_ + "[cp]"
    if int(order) < 10:
        regex_ = regex_ + '0' + order
    else:
        regex_ = regex_ + order
    return regex_


def main(oscaar_root, out_file):
    out_file = open(out_file, 'w')
    for dir in os.listdir(oscaar_root):
        transcript_iter = Path(oscaar_root + '/' + dir).glob('*materials.txt')
        for f in transcript_iter:
            transcript_file = str(f)
            transcript_name = transcript_file.split(".")[0].split("/")[-1]
            print(transcript_name)
            transcript_sub_name = ""
            if '_' in transcript_name:
                transcript_sub_name = transcript_name.split("_")[0]
            with open(transcript_file, 'r') as t:
                next(t)
                counter = 0
                for line in t:
                    line_splits = line.split(",")
                    list_ = line_splits[0]
                    order = line_splits[-1].strip("\n")
                    transcript = " ".join(line_splits[1:-1])
                    list_item = list_.split(" ")[1]
                    if ((dir == 'BEL') and (transcript_sub_name == 'BEL')) or (dir == 'BKB'):
                        audio_file_iter = Path(oscaar_root + "/" + dir + '/' + dir).glob(transcript_sub_name + \
                                                                                         get_regex_bel_bkb(list_item, order) + '.wav')
                    elif dir == 'IEEE':
                        audio_file_iter = Path(oscaar_root + "/" + dir + '/' + dir).glob(transcript_sub_name + \
                                                                                         get_regex_ieee(order) + '.wav')
                    elif dir == 'Clear Speech':
                        audio_file_iter = Path(oscaar_root + "/" + dir + '/' + dir).glob(transcript_sub_name + \
                                                                            get_regex_clear_speech(list_item, order) + '.wav')
                    else:
                        counter = counter + 1
                        audio_file_iter = Path(oscaar_root + "/" + dir + '/' + dir).glob(transcript_sub_name + "_" + \
                                                                                         '[A-Z][A-Z]' + str(counter) + '.wav')
                    for audio_file in audio_file_iter:
                        id_ = str(audio_file).split(".")[0].split("/")[-1]
                        out_file.write('oscaar_' + id_ + '\t' + str(audio_file) + '\t' + transcript + '\n')


if __name__ == '__main__':
    # needs two command line argument.
    # 1. root path of Oscaar
    # 2. output csv path
    main(sys.argv[1], sys.argv[2])















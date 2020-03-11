# Lexical Stress Detection - NN

Lexical stress detection using deep neural network.

ml-stress-detection-nn is a neural network architecture that identifies if a vowel phoneme
in an isolated word is stressed or un-stressed.
The poster gives a brief description of the overall process
[Project Poster](lexical-stress-detection/images/project_poster.pdf)

---
*To retrain the model follow the steps below:*
### Feature Extraction
##### Phoneme Alignment
The first step of feature extraction is forced phoneme alignment of audio files. Refer to the alignment
[readme](alignment/README.md).

Phoneme alignment needs the files in .wav format. If you've .flacc files, use 
[this script](scripts/convert_flac_to_wav.sh) to convert them to wav files.


##### Training Sample Generation
This process extracts spectral and non spectral features from the of each phoneme, stores them as numpy arrays and
writes to disk as .npy files. 

Since stress on a particular vowel phoneme is related to other vowel phonemes within that word, features of each 
phoneme is sandwiched between the features of preceding and succeeding phoneme.

For each phoneme two files are generated:
1. `*_mfcc.npy`: *Spectral features* - 13 MFCCs for 10 frames, their derivatives and double derivatives. This is
represented as a matrix of shape `13 x 30`. Preceding an succeeding phoneme features are added as channels, thus the
final shape of the matrix is `3 x 13 x 30`.
Refer to [`mfcc_extraction.py`](cnnmodel/feature_extraction/mfcc_extraction.py)
2. `*_other.npy`: *Non Spectral Features* - 6 non spectral features for the phoneme represented as a vector of length
6 which becomes a vector of length 18 after including preceding an succeeding phoneme features.
Refer to [`non_mfcc_extraction.py`](cnnmodel/feature_extraction/non_mfcc_extraction.py)

Sample generation is be done by running the [`sample_generation.py`](cnnmodel/feature_extraction/sample_genration.py)
script which takes three inputs as command line arguments:
1. Root path of the folder with wav files split into phonemes. This is the output of phoneme alignment.
2. Tab separated file with phoneme info. This is the csv generated by phoneme alignment.
3. Output path where npy files will be generated. Output will be split into three folders 0, 1 and 2 each with
unstressed, primary stress and secondary stress phoneme features. 
 
*Sample generation script is parallelized, CPU with 16 or more cores is recommended for running it.*

##### Class Balancing
After sample generation, primary stress phonemes were twice as much than unstressed. Secondary stress were a very 
small percentage of the total. For this project we completely ignored secondary stress and randomly sampled primary 
stress features approximately equal to unstressed.

##### Stop Words Removal
We removed features of [80 stop words](cnnmodel/feature_extraction/ignored_words.txt). Since the npy file names
have the word in them, a sample script can be written for this action.

##### Train Test Split
Use [`train_test_split.py`](cnnmodel/feature_extraction/train_test_split.py) to split data into train and val sets.
The script needs three input parameters as command line arguments.
1. Root path of the folder where npy sample files are stored. 
2. Train path
3. Test path
4. Test percentage - a floating point number int the range (0,1). We used 0.15.

### Model
The [`model`](cnnmodel/model.py) is a combination of CNN and DNN. Spectral features are fed into the CNN and the
non spectral into DNN. The output form these networks are concatenated and fed into another DNN and finally, the
softmax loss layer is used.

##### Training
The [`training.py`](cnnmodel/training.py) script takes five command line arguments:
1. Root path of train data
2. Root path of val data
3. Path where saved models are saved. If existing model checkpoint is present in this folder, training will
resume from that checkpoint.
4.Learning rate
5.Number of epochs

Hyper parameters like batch size can be changed in this script.
It also generates a file `data_check_test.csv` which has some info about predictions on the val set. This is useful
for debugging which samples are incorrectly classified. The five columns in the file are:
1. path: name with full path of the npy files
2. label: true label of the sample
3. pred: prediction by the model
4. prob_0: probability of predicting unstressed
5. prob_1: probability of predicting primary stress

Sample csv file:
```csv
path,label,pred,prob_0,prob_1
test/0/libri_5808-54425-0000_is_ih0_mfcc.npy,0,0,0.9996665716171265,0.00033342366805300117
test/1/libri_5808-54425-0000_years_ih1_mfcc.npy,1,1,6.26739677045407e-07,0.9999994039535522
test/1/libri_5808-54425-0000_five_ay1_mfcc.npy,1,1,2.3276076888123498e-07,0.9999997615814209
test/1/libri_5808-54425-0000_but_ah1_mfcc.npy,1,1,4.2122044874304265e-07,0.9999995231628418
```

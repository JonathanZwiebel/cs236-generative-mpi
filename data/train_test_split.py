# This file will split the data into train/test 
# This looks for downloaded transcode data in transcode and fetches the appropriate CSV files
# from BabyData10K

# After this is run reformat_data will need to be run

import os
import shutil

SEARCH_DIRS = ['../RealEstate10K/test', '../RealEstate10K/train']
TRANSCODED_DIRS = 'transcode'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
CSVFILE_DIR = 'csvfiles'

TRAIN_RATIO = 0.8

assert os.path.exists(TRANSCODED_DIRS)
counter = 1
for TRANS_DIR in os.listdir(TRANSCODED_DIRS):
    split = 'train'
    if counter % 5 == 0:
        split = 'test'

    shutil.copytree(TRANSCODED_DIRS + '/' + TRANS_DIR, split + '/' + TRANS_DIR)

    found = False
    for SEARCH_DIR in SEARCH_DIRS:
        if os.path.exists(SEARCH_DIR + '/' + TRANS_DIR):
            shutil.copyfile(SEARCH_DIR + '/' + TRANS_DIR, split + '/' + CSVFILE_DIR + '/' + TRANS_DIR)
            print('Moved ' + TRANS_DIR + ' in to ' + split)
            found = True
            break

    if not found:
        print("ERROR not found: " + TRANS_DIR)
        exit(0)

    counter += 1

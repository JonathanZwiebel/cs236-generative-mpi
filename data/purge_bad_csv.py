# To be run after reformat_data.py
# This checks all of the CSV files and removes the ones that have timestamps that correspond to data that is not in train or test

import os
import shutil

MASTER_DIRS = ['train', 'test']

for MASTER_DIR in MASTER_DIRS:
    no_folder_cnt = 0
    missing_timestamp_cnt = 0
    valid_cnt = 0
    
    for csv_file in os.listdir(MASTER_DIR + "/csvfiles"):
        f = open(MASTER_DIR + "/csvfiles/" + csv_file, 'r')
        yt_id = f.readline().split('=')[-1].strip()

        #print(yt_id)
        SEARCH_DIR = MASTER_DIR + "/" + yt_id + "/"
        #print(SEARCH_DIR)
        if not os.path.exists(SEARCH_DIR):
            no_folder_cnt += 1
            continue

        lines = f.readlines()[1:]
        valid_flag = True
        for line in lines:
            timestamp = line.split(' ')[0]
            if not os.path.exists(SEARCH_DIR + yt_id + "_" + timestamp + ".jpg"):
                valid_flag = False
                break
        if not valid_flag:
            missing_timestamp_cnt += 1
            os.remove(MASTER_DIR + "/csvfiles/" + csv_file)
            continue

        valid_cnt += 1

    print("\nFor " + MASTER_DIR)
    print("No folder: " + str(no_folder_cnt))
    print("Missing timestamp: " + str(missing_timestamp_cnt))
    print("Valid: " + str(valid_cnt))

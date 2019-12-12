# This script reformats our data from the output of the download script to the input for our model

import os
import shutil
import pickle

CSV_DIR = 'train/csvfiles/'
FILES_DIR = 'train/'

assert os.path.exists(CSV_DIR)

for csv_file in os.listdir(CSV_DIR):
    #dataset_id = loc[:-4]
    #print(dataset_id)
    f = open(CSV_DIR + csv_file, 'r')
    first = f.readline()
    yt_id = first.split('=')[1].strip()

    if not os.path.exists(FILES_DIR + yt_id):
        os.mkdir(FILES_DIR + yt_id)
    else:
        print("Folder for " + yt_id + " already exists")

    with open(FILES_DIR + csv_file + '/case.pkl', 'rb') as p_file:
        data = pickle.load(p_file)

    # ts_map goes from real to pose
    ts_map = {}
    for data_at_ts in data:
        if 'imgPath' in data_at_ts:
            ts_map[data_at_ts['imgPath'].split('/')[-1]] = data_at_ts['timeStamp']
            
    exact = 0
    nearby = 0
    for real_timestamp in os.listdir(FILES_DIR + csv_file):
        if real_timestamp in ts_map:
            shutil.copyfile(FILES_DIR + csv_file + "/" + real_timestamp, FILES_DIR + yt_id + '/' + yt_id + '_' + str(ts_map[real_timestamp]) + ".jpg")
            nearby += 1
        else:
            shutil.copyfile(FILES_DIR + csv_file + "/" + real_timestamp, FILES_DIR + yt_id + '/' + yt_id + '_' + real_timestamp)
            exact += 1

    print("finished for " + yt_id + " (" + csv_file + ") | real: " + str(exact) + " | fake:" + str(nearby))

'''
cd /var/www/git/NLP/Hotel-Reviews-REU/Vlad/predict_aspect_ratings/1_split_data/; python3 create_folds.py
'''

import sys;
import numpy as np;
import random;

DATA_SOURCE = "../0_data_source/derived_training_data.csv";
DATA_SOURCE_MOD = "data";
FOLDS = 3;
CREATE_VALIDATION_FOLD = True;

print("Loading data from " + DATA_SOURCE + " and sorting it into folds...");    
#########################################
## Load Input Data
#########################################
f = open(DATA_SOURCE, 'r');
lines = f.readlines();
del lines[0]; ## remove header line
random.shuffle(lines); ## shuffle lines
#lines = [str(i)+"\n" for i in range(100)];

#########################################
## Create validation / parameter-tuning fold if desired
#########################################
if(CREATE_VALIDATION_FOLD == True):
    validation_size = int(len(lines) / (FOLDS + 1));
    validation_lines = lines[:validation_size];
    del lines[:validation_size]; # remove validation lines from fold lines
    validation_fold = open("folds/" + DATA_SOURCE_MOD+".fold_validation", "w+");
    for line in validation_lines:
        validation_fold.write(line);
    validation_fold.close();
    
#########################################
## Create rest of folds
#########################################
## create files in which folds will be held
files = [];
for i in range(FOLDS):
    files.append(dict({"test" : open("folds/" + DATA_SOURCE_MOD+".fold_"+str(i)+".test", "w+"), "train" : open("folds/" + DATA_SOURCE_MOD+".fold_"+str(i)+".train", "w+")}));
## stick data in folds
items_per_fold = int(len(lines) / FOLDS);
for line_index, line in enumerate(lines):
    if(line.rstrip() == ""): continue;
    for file_index, a_file in enumerate(files):
        ## first 50 go in test of fold_1 if items per fold is 50, else go in train
        #print("new-----");
        #print("f_i : ", file_index);
        #print("l_i : ", line_index);
        #print("l_i = ", line_index, " -> ", np.floor(line_index / items_per_fold) );
        #line = str(line_index);
        test_class = int(np.floor(line_index / items_per_fold));
        #print("test_class : ", test_class);
        if(test_class == file_index):
            a_file["test"].write(line);
        else:
            a_file["train"].write(line);
f.close();
print("End.");
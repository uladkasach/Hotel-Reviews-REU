###############
## Import Libraries
###############
import tensorflow as tf
import sys
import numpy as np;
import random;
import datetime;

##########################################################################
## Return Regular Batch
##########################################################################
def return_regular_batch(data_source_path, batch_size, aspect_to_learn):
    global data_source_lines;
    global data_source_lines_index;
    
    ########################
    ## Initialize global variables if they do not exist
    ########################
    if 'data_source_lines' not in globals():
        data_source_lines = {
            "train" : None,
            "test" : None,
        }   
        data_source_lines_index = {
            "train" : 0,
            "test" : 0,
        }

    ######################
    ## Detect data source type
    ######################
    data_source_path = data_source_path[0];
    if ("test" in data_source_path.split(".")):
        this_source_type = "test";
    elif ("train" in data_source_path.split(".")):
        this_source_type = "train";
    else:
        print(" Data Source path is not a test or train type. Error.");
        exit();

    #######################
    ## Load this source data into memory if it is not already there
    #######################
    if(data_source_lines[this_source_type] is None):
        print("Loading ", this_source_type, " data");
        with open(data_source_path) as fp:
            source_lines = fp.readlines();
        random.shuffle(source_lines);
        data_source_lines[this_source_type] = source_lines;
    else:
        source_lines = data_source_lines[this_source_type];
        
    
    #######################
    ## Load settings
    #######################
    ## one hot depth, the output we will be predicting
    one_hot_depth = 1; ## one because this is regression
    
    ## batch size
    if(batch_size > 0):
        batch_data_length = batch_size;
    else:
        batch_data_length = len(source_lines);
        
        
    #######################
    ## extract data from input lines
    #######################
    keys_list = []; ## only supports one key per row atm
    y_data = np.zeros([batch_data_length, one_hot_depth], 'float');
    feature_data = None; #numpy.zeros([batch_size, len(feature_index)], 'float');
    
    ## Grab and parse data
    i = data_source_lines_index[this_source_type];
    while (len(keys_list) != batch_data_length):
        i += 1;
        if(i >= len(source_lines) - 1): ## if index exceeds amount of input, shuffle the lines for the next batch and reset index
            random.shuffle(source_lines);
            data_source_lines[this_source_type] = source_lines;
            i = 0;
            # print("Shuffling source_lines!");
            
        line = source_lines[i];
        
        parts = line.rstrip().split(", ");
        
        # define keys, labels, and features
            # ext_review_id, label_overall, label_location, label_service, label_price, location_neg, location_neu, location_pos, service_neg, service_neu, service_pos, price_neg, price_neu, price_pos
        #print(parts);
        this_key = parts[0];
        label_values = dict({
            "overall" : int(parts[1]),
            "location" : int(parts[2]),
            "service" : int(parts[3]),
            "price" : int(parts[4])
        })
        this_label_value = label_values[aspect_to_learn];
        these_features = np.array([float(j) for j in parts[5:]])

        # ensure that label is defined for this review (not all aspects are always rated by user)    
        if(int(this_label_value) == -1): continue; # skip if its not defined
            
        # append this key to keys list
        keys_list.append(this_key);
        
        # define row index based on keys_list
        this_row_index = len(keys_list) - 1;
        
        # set label to proper value
        y_data[this_row_index, 0] = float(this_label_value); 
        
        # append these features to feature matrix 
        if(feature_data is None):
            feature_data = np.zeros([batch_data_length, len(these_features)], 'float');
        feature_data[this_row_index, :] = these_features;

        #if(i == 0):
            #print("input 0 = ", this_key);
        
        

    data_source_lines_index[this_source_type] = i;
    
    #print(feature_data);
    #print(y_data);
    #print(keys_list);
    #exit();
    
    return feature_data, y_data, keys_list; 
    
    
    
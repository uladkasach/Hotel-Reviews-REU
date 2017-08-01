###########################################################
## Modules for Training
############################################################
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import time

import load_data
import save_data


#####################################################################################################################################
## Load Inputs and HPs
#####################################################################################################################################
#########################################################
## Read Arguments
#########################################################
arguments = dict();
acceptable_arguments = ['name', 'source_mod', 'rtrue',  'batch_size', 'learning_rate', 'n_hidden_1', 'n_hidden_2', 'epochs', 'classifier_choice', "validate", "aspect"];
if(len(sys.argv) < 2 or sys.argv[1] == "-h"):
    print ("Acceptable arguments : " + ", ".join(acceptable_arguments));
    print ("Required arguments : name, source_mod, aspect")
    print ("E.g., python3 classifier.py name:test source_mod:fold_0")
    exit();
for i in range(len(sys.argv)):
    if(i == 0):
        continue;
    this_argv = sys.argv[i];
    parts = this_argv.split(":");
    this_name = parts[0];
    this_value = parts[1];
    if(this_name not in acceptable_arguments):
        print(this_name, " is not an acceptable argument. Error.");
        exit();
    arguments[this_name] = this_value;
    

#########################################################
## Set Default Data
#########################################################
classifier_choice = "nn";
EPOCHS = 50000;
R_True = 1; ## weight of positive class
R_False = 1; ## weight of negative class
batch_size = 200;
learning_rate = 0.05;
n_hidden_1 = 30 # 1st layer number of features
n_hidden_2 = 5 # 2nd layer number of features
save_training = False;

#########################################################
## Update data to arguments
#########################################################
if('name' in arguments):
    delta_mod = arguments['name'];
else:
    print("name is required. Error.");
    exit();
if('source_mod' in arguments):
    source_mod = arguments['source_mod'];
    TRAIN_SOURCE = '../../1_split_data/folds/data.' + source_mod +'.train';
    TEST_SOURCE = '../../1_split_data/folds/data.' + source_mod +'.test';
    if('validate' in arguments and arguments['validate'] == "true"):
        TEST_SOURCE = '../../1_split_data/folds/data.fold_validation';
else:
    print("source_mod is required. Error.");
    exit();
if('aspect' in arguments):
    aspect_to_learn = arguments["aspect"];
else: 
    print("aspect is required. Error");
    exit();
if('rtrue' in arguments): R_True = float(arguments['rtrue']);
if('batch_size' in arguments): batch_size = int(arguments['batch_size']);
if('learning_rate' in arguments): learning_rate = float(arguments['learning_rate']);
if('n_hidden_1' in arguments): n_hidden_1 = int(arguments['n_hidden_1']);
if('n_hidden_2' in arguments): n_hidden_2 = int(arguments['n_hidden_2']);
if('epochs' in arguments): EPOCHS = int(arguments['epochs']);
if('classifier_choice' in arguments):  classifier_choice = (arguments['classifier_choice']);
if('save_training' in arguments and arguments['save_training'] == "true"): save_training = True;
    
    
    
#####################################################################################################################################
## Define Model Structure
#####################################################################################################################################
###################################
## Data Source Variables / Ops
###################################
feature_batch, label_batch, key_batch = load_data.return_regular_batch([TRAIN_SOURCE], batch_size, aspect_to_learn);
feature_count = feature_batch.shape[1];
label_count = label_batch.shape[1];

###################################
# Network Parameters
###################################
n_input = feature_count 
n_classes = label_count 
    
###################################
# tf Graph input
###################################
x = tf.placeholder(tf.float32, [None, feature_count]) ## Features
y = tf.placeholder(tf.float32, [None, label_count]) ## True Values


###################################
# Create model
###################################
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer    
    
    
###################################
# Store layers weight & bias
###################################
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


###################################
# Construct model
###################################
pred = multilayer_perceptron(x, weights, biases);
#########
# Define loss and optimizer
##########
#cost = tf.metrics.mean_squared_error(predictions=pred, labels=y);

squared_error = tf.square(tf.subtract(pred, y));
mean_squared_error = tf.reduce_mean(squared_error);
cost = mean_squared_error;
print(cost);
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


################################
## Define Evaluation Graph
################################
rounded_pred = tf.round(pred);
squared_error = tf.square(tf.subtract(rounded_pred, y));
mean_squared_error = tf.reduce_mean(squared_error);

display_pred = tf.transpose(rounded_pred);
display_label = tf.transpose(y);

###############################
## Define Initialization function for variables/ops
###############################
init = tf.global_variables_initializer() ## initialization operation




#####################################################################################################################################################
## Train and Classify
#####################################################################################################################################################
with tf.Session() as sess:
    sess.run(init);
    
    #################################################################
    ## Train Model
    #################################################################
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #for i in range(1200):
        # Retrieve a single instance:
    #print(col1);
    #print(sess.run([results]))
    #print(example);
    
    
    epochs = EPOCHS;
    display_ratio = 200;
    
    ## for tracking cost,acc per epoch
    training_progress = pd.DataFrame(columns =  ["epoch", "cost", "mean_squared_error"]);
    
    for i in range(epochs):
        batch_feature, batch_label, batch_key = load_data.return_regular_batch([TRAIN_SOURCE], batch_size, aspect_to_learn);
        
        if(i == 0):
            start_time = time.time()
            print ('Init Cost : ', end = '');
            print (sess.run(cost, feed_dict={x: batch_feature, y : batch_label}));
        
        if(i % (epochs/display_ratio) == 0 and i != 0):
            end_time = time.time();
            print ('Epoch %6d' % i, end = '');
            print(' ... cost : ', end = '');
            this_cost = (sess.run(cost, feed_dict={x: batch_feature, y : batch_label}));
            print ('%10f' % this_cost, end = '');
            print (',  mse : ', end = '');
            this_mse = (sess.run(mean_squared_error, feed_dict={x: batch_feature, y : batch_label}))
            print ('%10f' % this_mse, end = '');
            #print(' - lr : ', end = ''); 
            #print ('%10f' % sess.run(learning_rate), end = '');
            print (',  dt : ', end = '');
            delta_t = end_time - start_time;
            print ('%10f' % delta_t, end = '');
            print('');
            
            if(False): #use this to test that accuracy and mse are calculating correctly
                #print(sess.run(pred[0:10],  feed_dict={x: batch_feature, y : batch_label}));
                print(np.transpose(np.array(sess.run(rounded_pred[0:10],  feed_dict={x: batch_feature, y : batch_label}))));
                #print(batch_label[0:10]);
                print(np.transpose(np.array(sess.run(y[0:10],  feed_dict={x: batch_feature, y : batch_label}))));

            training_progress.loc[i] = [i, this_cost, this_mse];
            start_time = time.time()

        sess.run(train_step, feed_dict={x: batch_feature, y : batch_label})
        
    print ('Final Cost : ', end = '');
    final_cost_found = sess.run(cost, feed_dict={x: batch_feature, y : batch_label});
    print (final_cost_found);
    print ('Final Learning Rate : ', end = '');
    #print (sess.run(learning_rate));
    
    coord.request_stop()
    coord.join(threads)
    
    

    #########
    ## Testing Batch
    #########
    batch_feature, batch_label, batch_key = load_data.return_regular_batch([TEST_SOURCE], -1, aspect_to_learn);
    print ('Final Test Cost : ', end = '');
    final_cost_found = sess.run(cost, feed_dict={x: batch_feature, y : batch_label});
    print (final_cost_found);
    print ('Final MSE : ', end = '');
    final_mse_found = sess.run(mean_squared_error, feed_dict={x: batch_feature, y : batch_label});
    print (final_mse_found);
    
    
    '''
    classification_df = pd.DataFrame();
    classification_df["is_plant"] = np.array((batch_label[:, 1]), 'int');
    classification_df["pred_plant"] = max_predictions;
    classification_df["key"] = batch_key;
    classification_df["pred_0"] = predictions[:, 0];
    classification_df["pred_1"] = predictions[:, 1];
    save_data.save_classification(classification_df, delta_mod = delta_mod+'_test');

    #################################
    ## Save Hyperparameter config
    #################################
    epochs = EPOCHS;
    rtrue = R_True;
    hyperstring = "";
    hyperparamlist = ['delta_mod', 'source_mod',  'epochs',  'batch_size', 'learning_rate', 'n_hidden_1', 'n_hidden_2',  'rtrue' , 'final_cost_found', 'classifier_choice'];
    for name in hyperparamlist:
        name_of_var = name;
        val_of_var = eval(name);
        hyperstring += name_of_var + " : " + str(val_of_var) + "\n";

    myfile = open("results/"+delta_mod+"_z_hyperparams.txt", "w+");
    myfile.write(hyperstring);
    myfile.close();
    print("Hyperparameters written.");

    '''


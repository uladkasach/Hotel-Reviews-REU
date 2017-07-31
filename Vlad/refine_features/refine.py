import pickle;
import derive_features_and_labels;
import os;
import multiprocessing as mp;
import sys;

results_file = "results/derived_training_data.csv";
input_file = 'input/aspect_dictionary_1000.pkl'

## derive the features and record them to file
def derive_and_record(document):
    with open(results_file, 'a+') as the_file:
        result = derive_features_and_labels.derive(document);
        if(result == False): return "err"; ## handle errors
        data = result[1];
        the_file.write(data+"\n")
        return "written!";


if __name__ == '__main__':

    

    ## load input file
    with open(input_file, 'rb') as f:
         docs = pickle.load(f)


    ## write header and refresh the file
    with open(results_file, 'w+') as the_file:
        result = derive_features_and_labels.derive(docs[0]);
        header = result[0];
        data = result[1];
        the_file.write(header+"\n");

    #  removing processes argument makes the code run on all available cores
    pool = mp.Pool(processes=2)
    results = pool.map(derive_and_record, docs)
    print(results)
import pickle;
import derive_features_and_labels;
import os;
import multiprocessing as mp;
import sys;

input_mod = sys.argv[1];
bool_extend = False;
if(len(sys.argv) > 2 and sys.argv[2] == "--extend"): bool_extend = True;

results_file = "results/derived_training_data.csv";
input_file = 'input/'+input_mod+'.pkl'

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
    open_type = "w+";
    if(bool_extend): open_type = "a+"; # a+ does not truncate the file to size 0 before starting
    with open(results_file, open_type) as the_file:
        result = derive_features_and_labels.derive(docs[0]);
        header = result[0];
        data = result[1];
        the_file.write(header+"\n");

    #  removing processes argument makes the code run on all available cores
    pool = mp.Pool(processes=2)
    results = pool.map(derive_and_record, docs)
    print(results)
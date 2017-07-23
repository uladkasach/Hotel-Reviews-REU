import pickle;
import compare_score_for_review;


with open('input/aspect_dictionary_700.pkl', 'rb') as f:
     docs = pickle.load(f)
        
print(docs[0]);
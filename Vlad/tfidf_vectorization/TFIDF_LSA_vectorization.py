from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np;

input_file_path = "inputs/reviews.txt";
output_file_path = "results/vectors.txt";

############
## derive document set from input file
############
document_set = [];
i = 0;
with open(input_file_path) as f:
    for line in f:
        i += 1;
        line=line.decode('utf-8','ignore').encode("utf-8")
        parts = line.split(" ");
        text = " ".join(parts[1:]).rstrip();
        #print("the text is  : " + text);
        document_set.append(text);
        #print(i);
        #print(text);
        if(i > 1000): break;
        if(i % 1000 == 0): print("loading is at line " + str(i));
#document_set = ("The sky is blue.", "The sun is bright.", "The sun in the sky is bright.", "We can see the shining sun, the bright sun.")
document_count = i;

#################################################################
## Compute BOW (term frequency matrix)
#################################################################
print("Starting sklearn BOW counting...")
count_vectorizer = CountVectorizer(min_df=0.0000001) # 0.01 -> 1106 words, 0.000001 -> 41588, ~130000, 0.0000001 -> 
count_vectorizer.fit_transform(document_set)
#print "Vocabulary:", count_vectorizer.vocabulary_
vocabulary = count_vectorizer.vocabulary_;
reverse_dictionary = dict((v,k) for k, v in vocabulary.iteritems());
#print(reverse_dictionary);

# Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
freq_term_matrix = count_vectorizer.transform(document_set)
#print("Frequency Matrix:")
#print freq_term_matrix.todense()
print(freq_term_matrix.shape);

'''
print("Removing infrequent words...");
## get indecies of columns to drop
infrequent_word_indicies = [];
frequency_threshold = 0.01;
print("   `-> threshold : " + str(frequency_threshold));
vector_count = freq_term_matrix.shape[1];
print("   `-> all words count = " + str(vector_count))
for i in range(vector_count):
    this_sparce_column = freq_term_matrix[:, i];
    this_word_frequency = this_sparce_column.sum();
    #print(this_sparce_column);
    #print(this_word_frequency);
    #print(document_count);
    #print(this_word_frequency/float(document_count));
    if(this_word_frequency/float(document_count) < frequency_threshold): infrequent_word_indicies.append(i);
    
#print(infrequent_word_indicies);
print("   `-> frequent words count = " + str(vector_count - len(infrequent_word_indicies)));


## drop infrequent columns
def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()

## remove the infrequent columns
freq_term_matrix = dropcols_coo(freq_term_matrix, infrequent_word_indicies);
print(freq_term_matrix.shape)
'''

#################################################################
## Convert BOW to TF-IDF
#################################################################
print("Starting sklearn tfidf transformation...")
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

# print "IDF:", tfidf.idf_
# IDF: [ 0.69314718 -0.40546511 -0.40546511  0.        ]

tf_idf_matrix = tfidf.transform(freq_term_matrix)
# print tf_idf_matrix.todense()

#print(tf_idf_matrix);
#print(type(tf_idf_matrix));
print(tf_idf_matrix.shape);
#exit();


#################################################################
## Reduce Dimensionality of word vectors w/ truncated SVD (aka LSA) 
##      LSA is utilized instead of PCA to deal w/ sparce matrix
#################################################################
#https://roshansanthosh.wordpress.com/2016/02/18/evaluating-term-and-document-similarity-using-latent-semantic-analysis/
print("Reducing dimensionality w/ LSA...");
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 300)
lsa = svd.fit_transform(tf_idf_matrix.T).T

#print(lsa);
#print(type(lsa));
print(lsa.shape);
#exit();

#################################################################
## Output Vectors
#################################################################
print("outputting vectors")
vector_count = tf_idf_matrix.shape[1]; 
with open(output_file_path, "w+") as f:
    for i in range(vector_count):
        ##print("Now grabbing vector " + str(i))
        this_vector = lsa[:, i];
        this_vector_as_string = " ".join([str(float(value)) for value in this_vector])
        this_word = reverse_dictionary[i];
        output_line = this_word + " " + this_vector_as_string;
        if(i % 200 == 0): print("word at " + str(i) + " = " + this_word + ", dim = " + str(len(this_vector)));
        f.write(output_line.encode('ascii', 'ignore').decode('ascii')+"\n");
        
print("All done!");

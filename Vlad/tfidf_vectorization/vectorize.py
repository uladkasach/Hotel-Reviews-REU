from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
#document_set = ("The sky is blue.", "The sun is bright.", "The sun in the sky is bright.", "We can see the shining sun, the bright sun.")


count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(document_set)
#print "Vocabulary:", count_vectorizer.vocabulary_
vocabulary = count_vectorizer.vocabulary_;
reverse_dictionary = dict((v,k) for k, v in vocabulary.iteritems());
#print(reverse_dictionary);

# Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
freq_term_matrix = count_vectorizer.transform(document_set)
#print("Frequency Matrix:")
#print freq_term_matrix.todense()


tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

# print "IDF:", tfidf.idf_
# IDF: [ 0.69314718 -0.40546511 -0.40546511  0.        ]


tf_idf_matrix = tfidf.transform(freq_term_matrix)
# print tf_idf_matrix.todense()

dense_matrix = tf_idf_matrix.todense();
#print(type(dense_matrix));
print(dense_matrix.shape);

# [[ 0.         -0.70710678 -0.70710678  0.        ]
# [ 0.         -0.89442719 -0.4472136   0.        ]]

vector_count = dense_matrix.shape[1]; 
with open(output_file_path, "w+") as f:
    for i in range(vector_count):
        ##print("Now grabbing vector " + str(i))
        this_vector = dense_matrix[:, i];
        this_vector_as_string = " ".join([str(float(value)) for value in this_vector])
        this_word = reverse_dictionary[i];
        output_line = this_word + " " + this_vector_as_string;
        if(i % 200 == 0): print("word at " + str(i) + " = " + this_word);
        f.write(output_line+"\n");
        
print("All done!");

<h1>Multi-Aspect Sentiment Analysis Tool</h1>
<h2>Introduction</h2>
<p>This tool is a convenient wrapper for various different text analysis components that you can use to analyze text for specific topics you give it (using seed words), the sentiment of the text, and "rank" those text documents according to which have the strongest sentiment and topic presence.</p>

<h2>Data Type and Cleaning</h2>
<p>The data type used should be text where a line break represents a new document. This data can be cleaned using the built in "clean" method. You can use a custom file of stop words (if none is supplied, then the NLTK default set of wnglsh stop words is used) and stemmng is optional.</p>

<p>If a raw dataset is cleaned, then the new cleaned dataset (which is saved to disk) is automatically used for word2vec training.</p>

<h2>Training Word2Vec</h2>
<p>The only tool used to create word embeddings by this tool is gensim's version of word2vec. The user can supply the dimensionality of the output vectors, the number of children processes to use (for parallel computing), and whether to save the model to file (which can be loaded later for further training).</p>

<h2>Word Vectors</h2>
<p>Gensim's built in KeyedVectors object is very useful. You can load ones you've saved from a previous word2vec training or ones you have from elsewhere by using the "load_wv" method. The format for vector's supplied from elsewhere should be:
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vector_count vector_dimensions<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;word vector_values
</p>

<p>Example:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 3<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dog 0.4 2.0 0.9<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cat 4.0 0.4 1.4
</p>


<h2>Training KMeans</h2>
<p>Once word2vec is trained and we have our embeddings, we can train kmeans upon them. Use the "train_kmeans" method to train on a specific k, or "score_kmeans_range" to train on multiple k and score them. IMPORTANT: Scoring in this tool requires seed words. The "goodness" of a cluster is the cosine similarity of the mean vector of the seed words (of a specific aspect) and the mean vector of a cluster's words. These values are averaged from within a cluster (if more than one of the supplied topics is present in the same cluster) and then averaged across all clusters whose score is positive.</p>

<p>Also note that in order to train k means, you have to have uploaded all of the word vectors as one big matrix. This is easy if you run the "save_wv_matrix" method followed by the "load_wv_matrix" method.</p>

<h2>Retrofitting</h2>
<p>You can supply adjacency lists representing semantic relationships between words to the retrofitting tool to update the word embeddings already created and, hopefully, increase the scores. The choice of seed words is arbitrary and the results can be very sensitive to the choice. When you have files containing your seed words (one topic per file), then use the "create_seed_word_adjacency_list" method to build the adjacency list for you. You need this file for retrofitting.</p>

<h2>Seed Word Helpers</h2>
<p>Seed words can be tricky to find. A shortcut to find a base line set of words is to cluster the default word embeddings and use those clusters to pull in words similar to a keyword and, from this list, trim out the nonsense. In other words, you can supply a keyword to the "seed_word_helper" method and it will supply all of the words in the cluster to which that keyword belongs. Say you give it "coffee", then words similar to coffee (according to word2vec) will be supplied and you can trim out the ones you don't think are useful for your purpose. Run this for multiple keywords that are all part of your specific topic and build a robust seed word list. Note: If you don't trim anything out, then you're basically saying the default custering is good enough and there is no need to even try retrofitting.</p>

<h2>Scoring the Documents</h2>
<p>Once you've settled on a clustering and have a set of seed words per topic being considered, then you can score the orginal document set you started with. You need to supply the uncleaned raw dataset to the "build_aspect_dictionary" method, but you don't have to score the entire thing at once (will only score the numbr of documents you tell it to). If the beginning of a line in your raw text is a tag or id of some sort, then supply the start index of the word that begins the actual document using the "start_index" argument.</p>

<p>Scoring the documents requires two arguments: the "strength_of_presence" variable which only counts an aspect as being present in a sentence if the similarity of that sentence and the aspect seed words is above this number and a "penalize_long" argument that penalizes longer documents. By default these two are set to "0.5" and "false" respectively.

<h2>Final Thoughts</h2>
<p>This info may not be too detailed, but I did my best to add exceptions to the program to tell you when you're doing things wrong. Check out the code itself for some help and try running the program to get a feel for how to go about things.</p>
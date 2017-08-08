from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pickle
from sklearn.cluster import KMeans
import math
import operator
from os import system
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import codecs
import time
import os.path

class MASA(object):

	#initialization method
	def __init__(self,
		aspects = [],
		aspect_dictionary_file = None,
		clustered_words_file = None,
		seeds_adjacency_file = None,
		seed_word_files = [],
		raw_text = None,
		w2v_file = None,
		wv_file = None,
		wv_matrix_file = None,
	):

		object.__init__(self)

		#aspects topics
		self.set_aspects(aspects)

		#load the aspect dictionary
		self.load_aspect_dictionary(aspect_dictionary_file)

		#load the clustered words file
		self.load_clustered_words(clustered_words_file)

		#set the seed words adjacency file
		self.set_seeds_adjacency_file(seeds_adjacency_file)

		#set the seed word files
		self.set_seed_word_files(seed_word_files)
		
		#raw_text file 
		self.set_raw_text(raw_text)

		#load a model into memory if provided 
		self.load_w2v(w2v_file)
		
		#load keyed vectors into memory if provided 
		self.load_wv(wv_file)

		#load the X matrix if file provided 
		self.load_wv_matrix(wv_matrix_file)
		
	###############
	#Begin Getters#
	###############

	#get word vectors
	def get_wv(self):
		return self.__wv

	#get w2v model
	def get_w2v(self):
		return self.__w2v

	#get X matrix
	def get_X(self):
		return self.__X

	#get index2vocab
	def get_index2vocab(self):
		return self.__index2vocab

	#get vocab2index
	def get_vocab2index(self):
		return self.__vocab2index

	#get clustered words
	def get_clustered_words(self):
		return self.__clustered_words

	#get the aspects list
	def get_aspects(self):
		return self.__aspects

	#get seed word files
	def get_seed_word_files(self):
		return self.__seed_word_files

	#get aspect dicitonary
	def get_aspect_dictionary(self):
		return self.__aspect_dictionary

	#############
	#End getters#
	#############

	###########################
	#Begin Loaders and Setters#
	###########################

	#load aspect dicitonary from file
	def load_aspect_dictionary(self, file = None):

		if file != None:

			if not os.path.isfile(file):
				raise FileNotFoundError('Aspect dictionary file not found')
			elif file[-4:] != '.pkl':
				raise Exception('Must be a .pkl file')
			
			with open(file, 'rb') as f:
				self.__aspect_dictionary = pickle.load(f)

		else:
			
			self.__aspect_dictionary = None

	#load clustered words 
	def load_clustered_words(self, file = None):

		if file != None:

			if not os.path.isfile(file):
				raise FileNotFoundError('Clustered words file not found')
			elif file[-4:] != '.pkl':
				raise Exception('Must be a .pkl file')

			with open(file, 'rb') as f:
				self.__clustered_words = pickle.load(f)

		else:

			self.__clustered_words = None

	#load a word2vec model into memory 
	def load_w2v(self, file = None):

		if file != None:

			if not os.path.isfile(file):
				raise FileNotFoundError('word2vec file not found')

			self.__w2v = Word2Vec.load(file)

		else:

			self.__w2v = None

	#load a set of keyed vectors into memory 
	def load_wv(self, file = None, w2v_format_binary = False, w2v_format_text = False):

		if file != None:

			if not os.path.isfile(file):
				raise FileNotFoundError('Word vector file not found')

			#load in proper format
			if w2v_format_binary == True:
				self.__wv = KeyedVectors.load_word2vec_format(file, binary = True)
			elif w2v_format_text == True:
				self.__wv = KeyedVectors.load_word2vec_format(file, binary = False)
			else:
				self.__wv = KeyedVectors.load(file)

			#mappings
			self.__vocab2index = {}
			self.__index2vocab = {}

			#get index <--> vocab mappings
			for word in self.__wv.vocab:
				self.__vocab2index[word] = self.__wv.vocab[word].index
				self.__index2vocab[ self.__vocab2index[word] ] = word

		else:

			self.__vocab2index = {}
			self.__index2vocab = {}
			self.__wv = None

	#load word vectors as one big numpy matrix 
	def load_wv_matrix(self, file = None):

		if file != None:

			if not os.path.isfile(file):
				raise FileNotFoundError('Word vector matrix file not found')
			elif self.__wv == None:
				raise Exception('Need word vectors in memory to load matrix correctly. Run the \"load_wv\" function first.')

			print('Loading word vectors as matrix...', end = '', flush = True)

			t1 = time.time()

			#load the word vectors as a numpy matrix using numpy's loadtxt method
			self.__X = np.loadtxt(file, delimiter = ' ', skiprows = 1, converters = { 0: lambda w: self.__vocab2index[w.decode('utf-8')] })

			t2 = time.time()

			print('done')

			print('Matrix loaded in {} seconds.'.format(int(t2 - t1)))

		else:

			self.__X = None

	#set the aspects
	def set_aspects(self, aspects = []):
		
		if len(aspects) < 1:
			self.__aspects = None
		elif type(aspects) != type(list()):
			raise Exception('Aspects must be a list')
		
		self.__aspects = aspects

	#set the raw text file
	def set_raw_text(self, file = None):

		if file != None and not os.path.isfile(file):
			raise FileNotFoundError('Raw text file not found')

		self.__raw_text = file

	#set the adjacency fileame
	def set_seeds_adjacency_file(self, file = None):

		if file != None and not os.path.isfile(file):
			raise FileNotFoundError('Seeds adjacency file not found')

		self.__seeds_adjacency_file = file

	#set the seed word files
	def set_seed_word_files(self, files = []):

		if len(files) < 1:
			self.__seed_word_files = None
		elif type(files) == type(list()):
			for file in files:
				if not os.path.isfile(file):
					raise FileNotFoundError('File "{}" not found'.format(file))
		else:
			raise Exception('Seed word files must be submitted as a python list')
		
		self.__seed_word_files = files

	#########################
	#End Loaders amd Setters#
	#########################

	##############
	#Begin Savers#
	##############

	#save the aspect dicitonary to file
	def save_aspect_dictionary(self, out = 'Aspect_Dictionary/aspect_dictionary.pkl'):

		if out[-4:] != '.pkl':
			raise Exception('File name provided must be a .pkl file')
		elif self.__aspect_dictionary == None:
			raise Exception('No aspect dictionary in memory to save! Run "load_aspect_dictionary" first.')
		
		with open(out, 'wb') as f:
			pickle.dump(self.__aspect_dictionary, f, pickle.HIGHEST_PROTOCOL)

	#save the clustered words 
	def save_clustered_words(self, out = 'KMeans/clustered_words.pkl'):

		if self.__clustered_words == None:
			raise Exception('No clustered words in memory! Run train_kmeans first.')
		elif out[-4:] != '.pkl':
			raise Exception('Clustered words must be saved in a .pkl file')

		with open(out, 'wb') as f:
			pickle.dump(self.__clustered_words, f, pickle.HIGHEST_PROTOCOL)

	#save word vectors
	def save_wv(self, out = 'Word_Vectors/wv'):

		if self.__wv == None:
			raise Exception('No word vectors in memory')
		
		self.__wv.save(out)

	#save word vectors in standard word2vec output format 
	def save_wv_matrix(self, out = 'Pre_Clustering/wv_master.txt'):

		if self.__wv == None:
			raise Exception('No word vectors in memory to save! Run "load_wv" first.')
	
		print('Saving word vector matrix...', end = '', flush = True)
		
		t1 = time.time()
		
		with open(out, 'w') as f:

			#header
			f.write('{} {}\n'.format(len(self.__wv.vocab), len(self.__wv[ list(self.__wv.vocab.keys())[0] ])))

			#saves with word and its vector
			for word in self.__wv.vocab:
				f.write('{} {}\n'.format(word, ' '.join([ str(element) for element in self.__wv[word] ]) ) )

		t2 = time.time()

		print('done')

		print('Matrix file created in {} seconds'.format(int(t2 - t1)))

	############
	#End Savers#
	############

	################
	#Begin Trainers#
	################

	#train kmeans to data 
	def train_kmeans(self, k = 100, workers = 1, out = None):

		if type(self.__X) == type(None):
			raise Exception('No data to fit! Run "load_wv_matrix" first.')

		print('Fitting kmeans to data with k = {}...'.format(k), end = '', flush = True)

		t1 = time.time()

		kmeans = KMeans(n_clusters = k, random_state = 0, n_jobs = workers).fit(self.__X[:, 1:])

		t2 = time.time()

		print('done')

		print('KMeans fitted in {} minutes.'.format( int( (t2 - t1) / 60 ) ))

		#save if user requests it
		if os.path.isfile(out):

			if out[-4:] == '.pkl':
				with open(out, 'wb') as f:
					pickle.dump(kmeans, f, pickle.HIGHEST_PROTOCOL)
			
			else:
				raise Exception('Kmeans must be saved in a .pkl file')

		#organize the clustered words
		self.__clustered_words = [ [] for i in range(kmeans.get_params()['n_clusters']) ]
		for i, j in enumerate(kmeans.labels_):
			self.__clustered_words[j].append(self.__index2vocab[ self.__X[i, 0] ])

	#train the word2vec model on the raw text data 
	def train_word2vec(self, dimensions = 300, workers = 1, w2v_path = None, hold_w2v = False):

		#make sure a file is given
		if self.__raw_text == None:
			raise FileNotFoundError('No text data provided. Load text data using "set_raw_text".')

		#Helpful iterator, credit here: https://rare-technologies.com/word2vec-tutorial/
		class MySentences(object):
			
			def __init__(self, filename):
				self.filename = filename
		 
			def __iter__(self):
				for line in open(self.filename):
					yield line.rstrip().split()

		#initialize the iterator with the filename
		sentences = MySentences(self.__raw_text)

		print('Training Word2Vec with {} dimensions...'.format(dimensions), end = '', flush = True)

		t1 = time.time()

		#train the word2vec model
		w2v = Word2Vec(sentences, size = dimensions, workers = workers)

		t2 = time.time()

		print('done')

		print('Word2Vec trained in {} minutes.'.format(int( (t2 - t1) / 60 )))

		#save if requested
		if w2v_path != None:
			w2v.save(w2v_path)

		#keep the w2v in memory if requested
		if hold_w2v:
			self.__w2v = w2v

		#Get vectors only from the w2v
		self.__wv = w2v.wv

		#get index <--> vocab mappings
		for word in self.__wv.vocab:
			self.__vocab2index[word] = self.__wv.vocab[word].index
			self.__index2vocab[ self.__vocab2index[word] ] = word

	##############
	#End Trainers#
	##############

	#################
	#Begin Utilities#
	#################

	#clean the raw text file 
	def clean(self, stopfile = None, startindex = 0):

		#need a filename
		if self.__raw_text == None:
			raise FileNotFoundError('No raw text file provided')

		print('Cleaning raw text data...', end = '', flush = True)

		#rename class level cleaned text filename to one created here
		cleaned_text = '{}_cleaned.txt'.format(self.__raw_text[:-4])

		#initialize stopwords. default is nltk stopwords
		if stopfile == None:
			stops = set(stopwords.words('english'))
		elif os.path.isfile(stopfile):
			with codecs.open(stopfile, 'r', encoding = 'utf-8', errors = 'ignore') as f:
				stops = set([ word.strip().lower() for word in f.readlines() ])
		else:
			raise Exception('Stopfile not found')

		#this regex object will remove all punctuation
		regex = re.compile(r'[^a-zA-Z0-9\s]|[\_\^\`\[\]\\]', re.IGNORECASE)

		#clean the review file
		with codecs.open(self.__raw_text, 'r', encoding = 'utf-8', errors = 'ignore') as f:
			with open(cleaned_text, 'w') as cleaned:

				t1 = time.time()

				#go through every line in the file
				for line in f:

					#remove non-alphanumeric symbols
					line = regex.sub(' ', line)

					#split into tokens and ignore stopwords
					tokens = [ word.lower().strip() for word in line.split(' ') if word not in stops ]

					#remove empty elements from the list
					tokens = [ word for word in tokens if word != '' ]

					#ignore elements before start index
					tokens = tokens[ startindex: ]

					#write cleaned data to file
					if len(tokens) > 0:
						cleaned.write('{}\n'.format(' '.join(tokens)))

				t2 = time.time()

		#update the text file name
		self.__raw_text = cleaned_text

		print('done')

		#display time it took to do all of this
		print('Raw text cleaned in {} minutes.'.format(int( (t2 - t1) / 60 )))

	#create the seed word adjacency list from the seed word files provided 
	def create_seed_word_adjacency_list(self, out = 'Seeds/seeds.txt'):

		if self.__seed_word_files == None:
			raise Exception('Run "set_seed_word_files" first.')

		#ouput to single file
		with open(out, 'w') as o:

			#go through every file
			for name in self.__seed_word_files:

				with codecs.open(name, 'r', encoding = 'utf-8', errors = 'ignore') as f:

					lines = [ line.rstrip() for line in f.readlines() ]
					
					#each seed word needs to be at the head of its own adjacency list
					for i in range(len(lines)):

						o.write('{} '.format(lines[i]))

						for j in range(len(lines)):

							if i != j:

								o.write('{} '.format(lines[j]))

						o.write('\n')

	#run the retrofit program 
	def retrofit(self, program_path = 'Retrofit/retrofit.py', out = 'Pre_Clustering/wv_master.txt', iterations = 10):

		if self.__seeds_adjacency_file == None:
			raise Exception('Need adjacency list to retrofit.')
		elif self.__wv == None:
			raise Exception('Need the word vector object in memory')

		with open('temp.txt', 'w') as f:
			for word in self.__wv.vocab:
				f.write('{} {}\n'.format(word, ' '.join([ str(element) for element in self.__wv[word] ]) ))

		t1 = time.time()

		#run the retrofit program
		print('Running retrofit...')
		print('Retrofit complete; exit code: {}'.format(system('python {} -i temp.txt -l {} -n {} -o {}'.format(program_path, self.__seeds_adjacency_file, iterations, out))))

		t2 = time.time()

		print('Retrofit finished in {:.4f} minutes.'.format( (t2 - t1) / 60 ))

		system('rm temp.txt')

		#read retrofitted vectors into memory
		with open(out, 'r') as f:
			lines = [ line.rstrip() for line in f.readlines() ]

		#add the header back to the file
		with open(out, 'w') as f:
			f.write('{} {}\n'.format(len(self.__wv.vocab), len(self.__wv[ list(self.__wv.vocab.keys())[0] ])))
			for line in lines:
				f.write('{}\n'.format(line))

	#score varous kmeans
	def score_kmeans_range(self, ks = list(range(100, 1100, 100)), workers = 1, out = 'kscores.txt'):

		if type(ks) != type(list()):
			raise Exception('Must provide a python list of k values')
		elif type(self.__X) == type(None):
			raise Exception('Must have a data matrix to score')
		elif self.__seed_word_files == None:
			raise Exception('Must have files containing aspect seed words. Run "set_seed_word_files" first.')

		#This run's setup
		with open(out, 'a') as f:
			f.write('Word2Vec Dimensions: {}\n'.format(len(self.__wv[ list(self.__wv.vocab.keys())[0] ])))

		#check every k
		for k in ks:

			#train a kmeans
			self.train_kmeans(k = k, workers = workers, out = 'KMeans/kmeans_{}.pkl'.format(k))
			
			#save the clustered words
			self.save_clustered_words(out = 'KMeans/clustered_words_{}.pkl'.format(k))

			#score for this k
			kscore = 0.0
			numclusters = 0

			print('Scoring clusters with k = {}...'.format(k), end = '', flush = True)

			#score the clustering
			for clstr in self.__clustered_words:
				cluster_set = set(clstr)
				clusterscore = 0.0
				numaspects = 0

				#go through every aspect
				for filename in self.__seed_word_files:
					with open(filename, 'r') as f:

						#seed words
						temp = set([ line.rstrip() for line in f ])

						#don't worry about clusters that don't ahve any aspect seed words
						if len(temp & cluster_set) > 0:
							clusterscore += self.similarity(np.mean([ self.__wv[word] for word in temp & cluster_set ], axis = 0), np.mean([ self.__wv[word] for word in cluster_set ], axis = 0))
							numaspects += 1

				#only care about clusters with aspects
				if numaspects > 0:
					numclusters += 1
					kscore += clusterscore / numaspects

			#average score
			kscore /= numclusters

			#write score to file
			with open(out, 'a') as f:
				f.write('k = {}, {}\n'.format(k, kscore))

			print('done')

		#line break
		with open(out, 'a') as f:
			f.write('\n')

	#seed word creation helper 
	def seed_word_helper(self, keyword = None, out = 'helper.txt', trim = None):

		if keyword == None:
			raise Exception('You need to provide a keyword!')
		elif self.__clustered_words == None:
			raise Exception('Need clustered words. Run "load_clustered_words" first or run kmeans')
		elif self.__wv == None and trim:
			raise Exception('Need word vectors in memory to trim. Run "load_wv" first')

		#warning
		if len(self.__clustered_words) < 100:
			print('Warning: If k is too small, then chances are that you\'re going to have a LOT of words to sift through. Retrain kmeans with a larger k to address this.')

		#intialize set
		helpers = set()

		#pull current helpers into memory
		if os.path.isfile(out):
			with open(out, 'r') as f:
				helpers = set([ line.rstrip() for line in f.readlines() ])

		#add words
		with open(out, 'w') as f:

			#find the cluster
			for c in self.__clustered_words:
				if keyword in c:
					helpers |= set(c)

			#trim to the most frequent n words
			if type(trim) == type(100):
				trimmed = sorted([ item[0] for item in sorted([ (word, self.__wv.vocab[word].count) for word in helpers ], key = operator.itemgetter(1), reverse = True) ][:trim])

				#write to file
				for word in trimmed:
					f.write('{}\n'.format(word))

			else:

				#write to file
				for word in sorted(helpers):
					f.write('{}\n'.format(word))

	#cosine similairty function
	def similarity(self, u, v):

		return u.T.dot(v) / (np.sqrt(u.T.dot(u)) * np.sqrt(v.T.dot(v)))

	###############
	#End Utilities#
	###############

if __name__ == '__main__':

	

	"""
	#default vector scoring
	for n_dim in range(100, 600, 100):
	
		masa = MASA(raw_text = 'Train_Data/reviews_cleaned.txt', seed_word_files = ['Seeds/amenities.txt', 'Seeds/location.txt', 'Seeds/service.txt', 'Seeds/price.txt' ])
		masa.train_word2vec(workers = 4, dimensions = n_dim, w2v_path = 'Model/w2v_{}'.format(n_dim))
		masa.save_wv(out = 'Word_Vectors/wv_{}'.format(n_dim))
		masa.save_wv_matrix(out = 'Pre_Clustering/wv_master_{}.txt'.format(n_dim))
		masa.load_wv_matrix('Pre_Clustering/wv_master_{}.txt'.format(n_dim))
		masa.score_kmeans_range(workers = 4)

	#new section
	with open('kscores.txt', 'a') as f:
		f.write('Begin retrofitted vectors')
		f.write('-------------------------')

	#retrofitted vector scoring
	for n_dim in range(100, 600, 100):

		masa = MASA(wv_file = 'Word_Vectors/wv_{}'.format(n_dim), seeds_adjacency_file = 'Seeds/seeds.txt', seed_word_files = ['Seeds/amenities.txt', 'Seeds/location.txt', 'Seeds/service.txt', 'Seeds/price.txt' ])
		masa.retrofit(out = 'Pre_Clustering/retro_wv_master_{}.txt'.format(n_dim), iterations = 10000)
		masa.load_wv_matrix(file = 'Pre_Clustering/retro_wv_master_{}.txt'.format(n_dim))
		masa.score_kmeans_range(workers = 4)
	"""

"""
Add: 
	dictionary creation logic
"""
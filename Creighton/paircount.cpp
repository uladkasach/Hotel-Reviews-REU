#include <unordered_set>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <pthread.h>
#include <mutex>
#include <time.h>
#include <unordered_map>

//mutual exclusion lock
std::mutex mtx;

//combination formula
double nChoose2(unsigned n) {
    
    if (2 > n)
    	return 0;

    double result = n;
    for(int i = 2; i <= 2; ++i) {
        result *= (n - i + 1);
        result /= i;
    }

    return result;
} //end combination formula

//struct containing thread data
struct thread_data {
	
	size_t thread_id;

	std::vector<std::unordered_set<std::string>>* docs;
	std::vector<std::vector<std::string>>* top20s;
	std::unordered_map<std::string, int>* pairmap;

	int limits[2];

};

//runs in each thread
void* checkpairs(void* args) {

	//cast the arguments into usable types
	struct thread_data* data = (struct thread_data*)args;

	//this just makes the loop below less codey
	std::vector<std::unordered_set<std::string>>* docs = data->docs;
	std::vector<std::vector<std::string>>* top20s = data->top20s;
	std::unordered_map<std::string, int>* pairmap = data->pairmap;

	int left = data->limits[0];
	int right = data->limits[1];

	//do each cluster
	for(int b = 0; b < 100; b++) {

		//check each pair
		for(int i = 0; i < top20s->at(b).size(); i++) {
			for(int j = i + 1; j < top20s->at(b).size(); j++) {

				//check each doc
				for(int k = left; k < right; k++) {

					//only iterate the count if both words are in the set
					if(docs->at(k).find(top20s->at(b)[i]) != docs->at(k).end() && docs->at(k).find(top20s->at(b)[j]) != docs->at(k).end()) {

						//critical section: iterate counts
						mtx.lock();
						try {
							pairmap->at( top20s->at(b)[i] + "-" + top20s->at(b)[j] )++;
						}
						catch(const std::out_of_range& e) {
							pairmap->insert({ top20s->at(b)[i] + "-" + top20s->at(b)[j], 1 });
						}
						mtx.unlock();
					
					} //end if
				} //end for
			} //end for
		} //end for

	} //end for

	//exit the thread
	pthread_exit(NULL);

} //end paircounts

//main program
int main(int argc, char* argv[]) {

	std::vector<int> dims = { 50, 150, 200, 250, 300, 350, 400, 450, 550 };

	time_t start, stop;
	start = time(0);

	for(int d = 0; d < dims.size(); d++) {

		//which dimension are we on right now?
		std::string dim = std::to_string(dims[d]);

		//number of threads (for later)
		size_t NUM_THREADS = 4;

		//how many docs to check per go
		//picked a number which will leave no
		//remainder lines in main loop
		size_t basket = 979;

		//Word counts
		std::fstream wordcounts("../../Misc/vocabcount.txt", std::ios::in);
		std::unordered_map<std::string, int> countmap;

		//word pair map for each pairs gross count
		std::unordered_map<std::string, int> pairmap;

		//holder string for all of our loops
		std::string line;

		//make hashmap of the word and its doccount
		while(std::getline(wordcounts, line)) {

			std::string sub = line.substr(0, line.find(" "));
			line = line.substr(line.find(" ") + 1, line.length());
			countmap[sub] = atof(line.c_str());

		} //end while

		wordcounts.close();

		//Top 20s
		std::fstream top20("../../Pre_Paircounting/Top20_" + dim + "d.txt", std::ios::in);

		//top 20s
		std::vector<std::vector<std::string>> top20s(100);

		//load the wordbags into memory
		size_t i = 0;
		while(std::getline(top20, line)) {

			if(line == "###")
				i++;

			//add to the temp vector
			if(line != "###")
				top20s[i].push_back(line);

		} //end while

		top20.close();

		//Cleaned Reviews
		std::fstream cleaned("../../cleaned_reviews.txt", std::ios::in);

		//list of lines, each lne split into tokens
		std::vector<std::unordered_set<std::string>> docs(basket);

		//times
		time_t t1, t2, t3;
		t1 = time(0);

		//add all of the tokenized lines to the doc list
		i = 0;
		size_t ckct = 0;
		while(std::getline(cleaned, line)) {

			//holder set
			std::unordered_set<std::string> tempset;

			//put each token from the line in the temp vector
			size_t pos = 0;
			while(pos != std::string::npos) {
				pos = line.find(" ");
				std::string sub = line.substr(0, line.find(" "));
				line = line.substr(line.find(" ") + 1, line.length());
				tempset.insert(sub);
			} //end while

			//add the tokenized line to the master doc list
			docs[i++] = tempset;

			//we only want to a section at a time to save memory
			if(i == basket) {

				//our loading bar
				std::cout << "\r" << (int)(((float)ckct++ / (float)(2827352 / basket)) * 100) << "\% complete";
				std::cout.flush();

				//reset the i counter
				i = 0;

				//the amount of lines each thread will check
				int width = basket / NUM_THREADS;

				//threads setup
				pthread_t threads[NUM_THREADS];
				pthread_attr_t attr;
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				void* status;
				int rc;
				struct thread_data* td[NUM_THREADS];
				for(int j = 0; j < NUM_THREADS; j++)
					td[j] = new thread_data();

				//loop drives the thread calls
				for(int j = 0; j < NUM_THREADS; j++) {

					//set thread ids
					td[j]->thread_id = j;

					//pointer to top20s for each thread (reading only, so no threat of race condition)
					td[j]->top20s = &top20s;

					//pointer to the list of doc set (reading only, so no threat of race condition)
					td[j]->docs = &docs;

					//pointer to the pair map
					td[j]->pairmap = &pairmap;

					//split the amount of file to b checked by each thread
					td[j]->limits[0] = (j * width);
					if(j == NUM_THREADS - 1)
						td[j]->limits[1] = basket;
					else
						td[j]->limits[1] = ((j + 1) * width);

					//spawn thread
					rc = pthread_create(&threads[j], &attr, checkpairs, (void*)(td[j]));

					//in case thread creation didn't work
					if(rc) {
						std::cout << "Error: Unable to create thread " << rc << std::endl;
						exit(-1);
					} //end if

				} //end for

				//reset the attr
				pthread_attr_destroy(&attr);

				//wait for the threads to finish before continuing
				for(int j = 0; j < NUM_THREADS; j++) {

					//wait for thread j to finish
					rc = pthread_join(threads[j], &status);

					//error joining thread
					if(rc) {
						std::cout << "Error: unable to join " << rc << std::endl;
						exit(-1);
					} //end if

				} //end for

				//delete heap data
				for(int j = 0; j < NUM_THREADS; j++)
					delete td[j];

			} //end if

		} //end while

		//close the file
		cleaned.close();

		//now that everything is counted, lets find the probabilities

		//average probabilities
		std::vector<double> probs(100, 0);
		
		//go through each cluster
		for(i = 0; i < 100; i++) {

			//got through each pair
			for(int j = 0; j < top20s[i].size(); j++) {
				for(int k = j + 1; k < top20s[i].size(); k++) {

					std::string lw = top20s[i][j];
					std::string rw = top20s[i][k];
					double pc = (double)pairmap[lw + "-" + rw];
					
					//average of p(lw+rw|lw) and p(lw+rw|rw)
					probs[i] += ( pc / (double)countmap[lw] + pc / (double)countmap[rw] ) / 2;

				}
			}

			//find average of all the probabilities in the cluster
			probs[i] = probs[i] / nChoose2(top20s[i].size());

		}

		t2 = time(0);
		double hrs = difftime(t2, t1) / 60 / 60;
		int rh = (int)hrs;
		int rm = (int)((hrs - rh) * 60);
		int rs = (int)( (((hrs - rh) * 60) - (int)((hrs - rh) * 60)) * 60 );
		std::cout << std::endl << "Total runtime: " << rh << " hour(s), " << rm << " minutes, and " << rs << " seconds." << std::endl;
		std::cout.flush();

		//output results
		std::fstream results("../../Cluster_Metrics/pairprobs_" + dim + "d.txt", std::ios::out);

		//write the results file
		for(auto i : probs)
			results << i << std::endl;

		//stitch em up, we're done here
		results.close();

	}

	stop = time(0);
	double hrs = difftime(stop, start) / 60 / 60;
	int rh = (int)hrs;
	int rm = (int)((hrs - rh) * 60);
	int rs = (int)( (((hrs - rh) * 60) - (int)((hrs - rh) * 60)) * 60 );
	std::cout << std::endl << "TOTAL Total runtime: " << rh << " hour(s), " << rm << " minutes, and " << rs << " seconds." << std::endl;
	std::cout.flush();

	return 0;

} //end main
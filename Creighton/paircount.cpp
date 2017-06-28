//128516
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

//mutual exclusion lock
std::mutex mtx;

//struct containing thread data
struct thread_data {
	
	size_t thread_id;

	std::vector<std::unordered_set<std::string>>* docs;
	std::vector<std::vector<std::string>>* top20s;
	std::vector<int>* paircounts;

	int limits[2];

};

//runs in each thread
void* checkpairs(void* args) {

	//cast the arguments into usable types
	struct thread_data* data = (struct thread_data*)args;

	//this just makes the loop below less codey
	std::vector<std::unordered_set<std::string>>* docs = data->docs;
	std::vector<std::vector<std::string>>* top20s = data->top20s;
	std::vector<int>* paircounts = data->paircounts;
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
						paircounts->at(b)++;
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
int main() {

	//number of threads (for later)
	size_t NUM_THREADS = 4;

	//how many docs to check per go
	//picked a number which will leave no
	//remainder lines in main loop
	size_t basket = 979;

	//Top 20s
	std::fstream top20("Pre_Paircounting/Top20_600d.txt", std::ios::in);

	//top 20s
	std::vector<std::vector<std::string>> top20s(100);

	//load the wordbags into memory
	size_t i = 0;
	std::string line;
	size_t outct = 1;
	while(std::getline(top20, line)) {

		if(line == "###")
			i++;

		//add to the temp vector
		if(line != "###")
			top20s[i].push_back(line);

	} //end while

	top20.close();

	//Cleaned Reviews
	std::fstream cleaned("cleaned_reviews.txt", std::ios::in);

	//list of lines, each lne split into tokens
	std::vector<std::unordered_set<std::string>> docs(basket);

	//actual paircounts of each cluster
	std::vector<int> paircounts(100, 0);

	//times
	time_t t1, t2;
	t1 = time(0);

	//add all of the tokenized lines to the doc list
	i = 0;
	size_t ckct = 1;
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
			std::cout << "\r" << ((float)ckct++ / (float)(2827352 / basket)) * 100 << "\% complete.";
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

				//pointer to pairwise counts (writing, so mutex will be used in function above)
				td[j]->paircounts = &paircounts;

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

	//display how long everything took
	t2 = time(0);
	std::cout << std::endl << difftime(t2, t1) / 60 / 60 << std::endl;
	std::cout.flush();

	//close the file
	cleaned.close();

	//output results
	std::fstream results("Cluster_Metrics/paircounts_600d.txt", std::ios::out);

	//write the results file
	for(auto i : paircounts)
		results << i << std::endl;

	//stitch em up, we're done here
	results.close();

	return 0;

} //end main
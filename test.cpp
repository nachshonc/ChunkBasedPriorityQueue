//  Authors: Nachshon Cohen, Anastasia Braginsky.
//  Code for testing the Chunked Based Priority Queue, CBPQ.
//  The algorithmic description appears in the paper:
//  "CBPQ: High Performance Lock-Free Priority Queue",
//  Anastasia Braginsky, Nachshon Cohen and Erez Petrank, EuroPar'16
//
//  The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED.


// #include <chrono>
#include <time.h>
#include <stdlib.h>     /* abs */
#include <algorithm>	/* sort */
#include <math.h>       /* pow */
#include <string.h>		/* memset */
#include <unistd.h>
#include <limits.h>

#include "test.h"

#include "debug.h"
#include "LinkedList.h"
#include "rand.h"
#include "globals.h"
#include "Atomicable.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//#define MAC

int 	numOfThreads	= 1;
float 	sleepTime 		= 1;		// seconds for throughput
unsigned int sizeForTest = 0;

__thread unsigned long nextr = 1;
volatile static bool run 	= false;	// used for synchronization of all thread starts

int testSingleThread();					// forward declaration for single thread testing
void preliminaryTesting();				// forward declaration for preliminaries testing

/***DES***/
/* preload array with exponentially distanced integers for the
 * DES workload */
#define EXPS 100000000
unsigned long *exps;
int exps_pos = 0;
void gen_exps(unsigned long *arr, gsl_rng *rng, int len, int intensity);
bool des_workload = false;
/* generate array of exponentially distributed variables */
void
gen_exps(unsigned long *arr, gsl_rng *rng, int len, int intensity)
{
	int i = 0;
	arr[0] = 2;
	while (++i < len){
		arr[i] = arr[i-1] + simRandom();
		//     (gsl_ran_geometric (rng, 1.0/(double)intensity));
		arr[i] = arr[i]%(INT_MAX/2);
	}
}
/***DES***/

/********************************************************************************************************/
// diff - function for supporting time management, when using clock_gettime()
timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}
/********************************************************************************************************/




/********************************************************************************************************/
// thread_deletes - per thread function that tries to delete elements from the Priority Queue (PQ) till 
// it is empty or till run-flag stops this thread. This method uses "values" array only when in debug mode.
void* thread_deletes(void *data){

	ThrInf *info 				= (ThrInf*)data;
	bool throughput				= info->throughput;
	ChunkedPriorityQueue *pq 	= info->pq;
	info->deleteCnt				= 0;
	DEB(int *values 			= info->values;)	// values array includes indications for all elements of
	int key = 0, iter = 0;							// PQ, and it is used for validation

	while(!run){__sync_synchronize();pthread_yield();}// wait till all threads are dispatched   

	PRINT("Thread %d starts to delete in PQ, for throughput?:%d\n", info->id, (int)throughput);

	if (throughput) {
		do{
			DEB(if(++iter>100000000)	{printf("InfLoop\n");assert(0);})

			key = pq->delmin(info);	info->deleteCnt++;	// delete minimum

			DEB(if(key>=0){								// sanity check, used for debugging
				if(values[key]==0){
					printf("Thread %d: Error, key %d deleted without inserting it first. \n", info->id, key);
					assert(0);
				}
				__sync_fetch_and_sub(&values[key], 1);
			})

		} while ( run );								// stop when sleepTime seconds passed
	} else {
		do{
			DEB(if(++iter>100000000){printf("InfLoop\n");	assert(0);})

			key = pq->delmin(info);	info->deleteCnt++;	// delete minimum

			DEB(if(key>=0){								// sanity check, used for debugging
				if(values[key]==0){
					printf("Thread %d: Error, key %d deleted without inserting it first. \n", info->id, key);
					assert(0);
				}
				__sync_fetch_and_sub(&values[key], 1);
			})

		} while ( key != -1 );							// when PQ is empty it returns -1 on delete
	}

	PRINT("Thread %d has finished deletions\n", info->id);
	return NULL;
}
/********************************************************************************************************/



/********************************************************************************************************/
// thread_mixed - per thread function that tries to delete/insert elements from/to the Priority Queue (PQ)
// randomly. Approximately 50% insertions and 50% deletions. Works for specific number of operations or till
// run-flag stops this thread.
void* thread_mixed(void *data){

	ThrInf *info 				= (ThrInf*)data;
	ChunkedPriorityQueue *pq 	= info->pq;
	info->deleteCnt				= 0;
	info->insertCnt				= 0;
	int key  = 0, 	iter = 0;
	simSRandom(info->id*info->id);   

	while(!run){__sync_synchronize();pthread_yield();}// wait till all threads are dispatched    
	PRINT("Thread %d starts to delete and to insert in PQ\n", info->id);

	do{
		DEB(if(++iter>100000000) assert(0);)				// protection against infinite loop 100 000 000
		key = 1 + simRandom()%(INT_MAX/2);

		if (key % 2 == 0)  {						// in half of the cases do deletion

			key = pq->delmin(info);				// delete minimum
			info->deleteCnt++;
			DEB(
					if (info->deleteCnt % 10000 == 0)
						PRINT2("Thread %d deleted 10000 keys and now deleted minimum: %d\n", info->id, key);)

		} else {									// in other cases do insertion

			if (des_workload) {
				int pos = __sync_fetch_and_add(&exps_pos, 1);
				key = exps[pos];
			}
			else
				key = 1 + simRandom()%(INT_MAX/2);
			pq->insert(key, info);
			info->insertCnt++;
			DEB(
					if (info->insertCnt % 10000 == 0)
						PRINT2("Thread %d inserted 10000 keys and currently inserted %d\n", info->id, key);)

		}

	} while ( run );

	PRINT("Thread %d has finished mixed benchmark\n", info->id);
	return NULL;
}
/********************************************************************************************************/



/********************************************************************************************************/
// thread_mixed_80d_20i - per thread function that tries to delete/insert elements from/to the Priority Queue (PQ)
// randomly. Approximately 20% insertions and 80% deletions. Works for specific number of operations or till
// run-flag stops this thread.
void* thread_mixedi_80d_20i(void *data){

	ThrInf *info                                = (ThrInf*)data;
	ChunkedPriorityQueue *pq    = info->pq;
	info->deleteCnt                         = 0;
	info->insertCnt                         = 0;
	int key  = 0, op = 0, iter = 0;
	simSRandom(info->id*info->id);

	while(!run){__sync_synchronize();pthread_yield();}// wait till all threads are dispatched
	PRINT("Thread %d starts to delete and to insert in PQ\n", info->id);

	do{
		DEB(if(++iter>100000000) assert(0);)
		op = simRandom()%(100);

		if ( op < 20 )  {                                            // in 80% of the cases do deletion

			key = pq->delmin(info);                         // delete minimum
			info->deleteCnt++;
			DEB(
					if (info->deleteCnt % 10000 == 0)
						PRINT2("Thread %d deleted 10000 keys and now deleted minimum: %d\n", info->id, key);)

		} else {                                                                        // in other cases do insertion

			if (des_workload) {
				int pos = __sync_fetch_and_add(&exps_pos, 1);
				key = exps[pos];
			}
			else
				key = 1 + simRandom()%(INT_MAX/2);
			pq->insert(key, info);
			info->insertCnt++;
			DEB(
					if (info->insertCnt % 10000 == 0)
						PRINT2("Thread %d inserted 10000 keys and currently inserted %d\n", info->id, key);)

		}

	} while ( run );

	PRINT("Thread %d has finished mixed benchmark\n", info->id);
	return NULL;
}
/********************************************************************************************************/




/********************************************************************************************************/
// thread_inserts - per thread function that tries to inserts num_ops elements to the Priority Queue (PQ) 
// This method uses "values" array only when in debug mode.
void* thread_inserts(void *data){

	ThrInf *info 				= (ThrInf*)data;
	bool throughput				= info->throughput;
	ChunkedPriorityQueue *pq 	= info->pq;
	int num_ops             	= info->num_ops;
	DEB(int *values 			= info->values;)	// values array includes indications for all elements
	info->insertCnt				= 0;				// that are going to be inserted into the PQ

	while(!run){__sync_synchronize();pthread_yield();}// wait till all threads are dispatched
	PRINT2(" Thread %d starts to insert to PQ \n", info->id);


	unsigned int key  = 0, iter = 0;

	if (throughput) {
		do{
			DEB(if(++iter>100000000) assert(0);)

			if (des_workload) {
				int pos = __sync_fetch_and_add(&exps_pos, 1);
				key = exps[pos];
			} else
				key = 1 + simRandom()%(INT_MAX/2);

			pq->insert(key, info);
			info->insertCnt++;


			DEB(__sync_fetch_and_add(&values[key], 1);)	// no additional synchronization for times measuring
		} while ( run );
	} else {
		do{
			DEB(if(++iter>100000000){printf("InfLoop\n");assert(0);})

			if (des_workload) {
				int pos = __sync_fetch_and_add(&exps_pos, 1);
				key = exps[pos];
			}
			else
				key = 1 + simRandom()%(INT_MAX/2);

			pq->insert(key, info);
			info->insertCnt++;


			DEB(__sync_fetch_and_add(&values[key], 1);)	// no additional synchronization for times measuring
		} while ( --num_ops > 0 );
	}


	PRINT2("Thread %d has finished insertions\n", info->id);
	return NULL;
}
/********************************************************************************************************/

/********************************************************************************************************/
int insertSequentially(ChunkedPriorityQueue *pq, int *values) {

	int insCnt 	= 0;		
	ThrInf 	info	= {0};
	info.pq			= pq;
	DEB(info.values		= values;)
	info.num_ops  	= sizeForTest;
	info.id			= 70;

	// initialize the PQ sequentially, with increasing keys, and without repetitions
	for(unsigned int i=1; i<sizeForTest; ++i){
#ifdef DEBUG		
		int rand = abs((int)simRandom());	// get random number without lock, always returns same numbers
		values[i]=rand%2;					// update the values array for the future validation
		if(values[i]) {
			pq->insert(i, &info); 
			if (i == 16) printf("16 was inserted!\n");
			insCnt++;
		}
#else	
		pq->insert(i, &info); 
		insCnt++;
#endif	// DEBUG

	} // end of for loop

	DEB(pq->assertStructure();)			// now priority queue contains approximately sizeForTest/2 unique items

	return insCnt;
}
/********************************************************************************************************/

/********************************************************************************************************/
// testDeletesAndMixed - main function that dispatches the threads and tests the concurrent deletes 
// 						or deletes and inserts in PQ
double testDeletesAndMixed(ChunkedPriorityQueue *pq, bool onlyDeletes, int* values){
	int deletions = 0, insInFirst = 0, insertions = 0, eleminated = 0;
	pthread_t 	threads[numOfThreads];
	ThrInf 		infos[numOfThreads];
	memset(threads, 0, sizeof(pthread_t)*numOfThreads);
	memset(infos, 	0, sizeof(ThrInf)*numOfThreads);

	_assert( values!=NULL );

	for(int i=0; i<numOfThreads; ++i){	// dispatch the threads, to delete from pq till it is empty
		infos[i].pq			= pq;
		DEB(infos[i].values	= values;)
		infos[i].num = numOfThreads;
		infos[i].num_ops	= sizeForTest/numOfThreads;
		infos[i].id			= i;
		infos[i].throughput = true;
		if (onlyDeletes) pthread_create(&threads[i], NULL, &thread_deletes, &infos[i]);
		else			 pthread_create(&threads[i], NULL, &thread_mixed, &infos[i]);
	}
	printf("Stage II start: %d threads where lunched, only deletes? %d\n", numOfThreads, (int)onlyDeletes);

#ifndef MAC
	timespec time1, time2;				// start the time measurement
	clock_gettime(CLOCK_MONOTONIC, &time1);
#else
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
#endif

	run = true;							// let the threads go
	usleep( 1000000*sleepTime );		// sleep for sleepTime seconds
	run = false;						

	for(int i=0; i<numOfThreads; ++i){
		pthread_join(threads[i], NULL);
		deletions += infos[i].deleteCnt;
		insertions += infos[i].insertCnt;
		insInFirst += infos[i].insInFirstCnt;
		eleminated += infos[i].eleminCnt;
	}

	run = false;	

#ifdef MAC
	chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();
	printf("PriorityQ: time = %lld nanoseconds\n",
			std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
	printf("%lld = PriorityQ_: time in microseconds\n",
			std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
#else
	clock_gettime(CLOCK_MONOTONIC, &time2);	// end the total time timer
	double totalTime = (double)(diff(time1,time2).tv_sec) + ((double)(diff(time1,time2).tv_nsec) / 1E9);
	if (onlyDeletes)
		printf("%d threads, Chunked PQ DELETIONS: total time = %1.8f seconds, throughput: %d\n", 
				numOfThreads, totalTime, deletions+insertions);
	else
		printf("%d threads, Chunked PQ MIXED: total time = %1.8f seconds, throughput: %d, "
				"from them insertions to first: %d, from them eliminated: %d\n",
				numOfThreads, totalTime, deletions+insertions, insInFirst, eleminated);

#endif

	DEB(
			if (onlyDeletes)
				for(unsigned int i=0; i<sizeForTest; ++i)
					if(values[i]!=0) {
						printf("Error: value %d was not deleted (after delete throughput workload) \n", i);
						assert(0);
					}
	)

	return totalTime;
}
/********************************************************************************************************/




/********************************************************************************************************/
// test80Del20Ins - the function that dispatches the threads and tests the concurrent  
//                                             80% deletes and 20% inserts in PQ
double test80Del20Ins(ChunkedPriorityQueue *pq, bool onlyDeletes, int* values){

	int deletions = 0, insInFirst = 0, insertions = 0, eleminated = 0;
	pthread_t       threads[numOfThreads];
	ThrInf              infos[numOfThreads];
	memset(threads, 0, sizeof(pthread_t)*numOfThreads);
	memset(infos,   0, sizeof(ThrInf)*numOfThreads);

	_assert( values!=NULL );

	for(int i=0; i<numOfThreads; ++i){  // dispatch the threads, to delete from pq till it is empty
		infos[i].pq                     = pq;
		DEB(infos[i].values     = values;)
		infos[i].num = numOfThreads;
		infos[i].num_ops        = sizeForTest/numOfThreads;
		infos[i].id                     = i;
		infos[i].throughput = true;
		if (onlyDeletes) pthread_create(&threads[i], NULL, &thread_deletes, &infos[i]);
		else                     pthread_create(&threads[i], NULL, &thread_mixedi_80d_20i, &infos[i]);
	}
	printf("Stage II start: %d threads where lunched, only deletes? %d\n", numOfThreads, (int)onlyDeletes);

#ifndef MAC
	timespec time1, time2;                          // start the time measurement
	clock_gettime(CLOCK_MONOTONIC, &time1);
#else
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
#endif

	run = true;                                                 // let the threads go
	usleep( 1000000*sleepTime );            // sleep for sleepTime seconds
	run = false;

	for(int i=0; i<numOfThreads; ++i){
		pthread_join(threads[i], NULL);
		deletions += infos[i].deleteCnt;
		insertions += infos[i].insertCnt;
		insInFirst += infos[i].insInFirstCnt;
		eleminated += infos[i].eleminCnt;
	}

	run = false;

#ifdef MAC
	chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();
	printf("PriorityQ: time = %lld nanoseconds\n",
			std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
	printf("%lld = PriorityQ_: time in microseconds\n",
			std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
#else
	clock_gettime(CLOCK_MONOTONIC, &time2); // end the total time timer
	double totalTime = (double)(diff(time1,time2).tv_sec) + ((double)(diff(time1,time2).tv_nsec) / 1E9);
	if (onlyDeletes)
		printf("%d threads, Chunked PQ DELETIONS: total time = %1.8f seconds, throughput: %d\n",
				numOfThreads, totalTime, deletions+insertions);
	else
		printf("%d threads, Chunked PQ MIXED 80d/20i: total time = %1.8f seconds, throughput: %d, "
				"from them insertions to first: %d, from them eliminated: %d\n",
				numOfThreads, totalTime, deletions+insertions, insInFirst, eleminated);

#endif

	DEB(
			if (onlyDeletes)
				for(unsigned int i=0; i<sizeForTest; ++i)
					if(values[i]!=0) {
						printf("Error: value %d was not deleted (after delete throughput workload) \n", i);
						assert(0);
					}
	)

	return totalTime;
}
/********************************************************************************************************/





/********************************************************************************************************/
// testInserts - main function that dispatches the threads and tests the concurrent inserts in PQ
double testInserts(ChunkedPriorityQueue *pq){
	/*Debugging code*/
	DEB(int *values = new int[sizeForTest];
		assert( values!=NULL );
		int insCnt = 0;

		ThrInf 	info	= {0};
		info.pq			= pq;
		info.values		= values;
		info.num_ops  	= sizeForTest;
		info.id			= 71;

		values[0] = 0;						// initialize the PQ sequentially, with increasing keys, and without
		for(unsigned int i=1; i<sizeForTest; ++i){	// repetitions
			int rand =
					abs((int)simRandom());			// get random number without lock, always returns the same numbers
			values[i]=rand%2;				// update the values array for the future validation
			if(values[i]) {
				pq->insert(i, &info);
				insCnt++;
			}
		}
		pq->assertStructure();				// now PQ contains approximately sizeForTest/2 unique items
		printf("PQ with the parallel array checks and with %d elements is ready.... GO!\n", insCnt);
	)/*End debugging code. */

	pthread_t 	threads[numOfThreads];
	ThrInf 		infos[numOfThreads];
	memset(threads, 0, sizeof(pthread_t)*numOfThreads);
	memset(infos, 	0, sizeof(ThrInf)*numOfThreads);

	for(int i=0; i<numOfThreads; ++i){	// dispatch the threads
		infos[i].pq			= pq;
		DEB(infos[i].values	= values;)
		infos[i].num_ops	= sizeForTest/numOfThreads;
		infos[i].id			= i;
		infos[i].throughput = true;
		pthread_create(&threads[i], NULL, &thread_inserts, &infos[i]);
	}


#ifndef MAC
	timespec time1, time2;				// start the time measurement
	clock_gettime(CLOCK_MONOTONIC, &time1);
#else
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
#endif

	run = true;	__sync_synchronize(); 			// let the threads go
	usleep( 1000000*sleepTime );				// sleep for sleepTime seconds
	run = false;						
	int inserts = 0, insInFirst = 0;

	for(int i=0; i<numOfThreads; ++i){
		pthread_join(threads[i], NULL);
		inserts		+= infos[i].insertCnt;
		insInFirst	+= infos[i].insInFirstCnt;
	}

	run = false;

#ifdef MAC
	chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();
	printf("PriorityQ: time = %lld nanoseconds\n",
			std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
	printf("%lld = PriorityQ_: time in microseconds\n",
			std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
#else
	clock_gettime(CLOCK_MONOTONIC, &time2);	// end the total time timer
	double totalTime = (double)(diff(time1,time2).tv_sec) + ((double)(diff(time1,time2).tv_nsec) / 1E9);
	printf("%d threads, Chunked PQ INSERTS: total time = %1.8f seconds, throughput in a second: %d, from them to first: %d\n", 
			numOfThreads, totalTime, inserts, insInFirst);
	//printf("%ld = PriorityQ_: time in microseconds\n", (diff(time1,time2).tv_nsec / 1000));
#endif


	DEB(		
			for(unsigned int i=0; i<sizeForTest; ++i)
				if(values[i]!=0) {
					printf("Error: value %d was not deleted (after insertions)\n", i);
					assert(0);
				}

	)

	PRINT("STAGE II: Insertion execution finished successfully\n");

	return totalTime;
}
/********************************************************************************************************/



/********************************************************************************************************/
// testInsertsNoMeasurements - a method that dispatches 24 threads, for concurrent random insertions of 
// sizeForTest elements. A help method for further benchmark. 
void testInsertsNoMeasurements(ChunkedPriorityQueue *pq, int* values){

	int concurNum = 24;

	pthread_t 	threads[concurNum];
	ThrInf 		infos[concurNum];
	DEB(int	inserts = 0; int valCnt = 0;)

	for(int i=0; i<concurNum; ++i){		// dispatch the threads
		infos[i].pq		= pq;
		DEB(infos[i].values	= values;)
		infos[i].num_ops= sizeForTest/concurNum;
		infos[i].id		= i;
		infos[i].throughput = false;
		pthread_create(&threads[i], NULL, &thread_inserts, &infos[i]);
	}

	printf("%d threads where lunched for initial insertion of %d keys\n", concurNum, sizeForTest);
	run = true;__sync_synchronize();						// let the threads go

	for(int i=0; i<concurNum; ++i){
		pthread_join(threads[i], NULL);
		DEB(inserts		+= infos[i].insertCnt;)
	}

	DEB(
		for (unsigned int i=0; i<sizeForTest; ++i) valCnt += values[i];
		if (inserts != valCnt)
			PRINT("Number of insertions into PQ is %d, and number of debugging structure updates is %d\n",
				inserts, valCnt);
		_assert(inserts == valCnt);
	)

	run = false; 					// reset for future use
	printf("STAGE I: Insertions finished successfully\n");
}
/********************************************************************************************************/






/////////////////////////////////////////////////////////     Linked-List test    ////////////////////////////////////////////////////////////////
/********************************************************************************************************/
// threads_ll_del - per thread function that tries to delete elements from the Linked List (LL) till it 
// is empty (exactly the same as thread_deletes)
void* threads_ll_del(void *data){

	ThrInf *info = (ThrInf*)data;
	LinkedList *ll = (LinkedList*)info->pq;
	int *values = info->values;

	while(!run){__sync_synchronize();pthread_yield();}// wait till all threads are dispatched
	PRINT(" Thread %d starts to delete in LL\n", info->id);

	int key;
	int iter=0;

	do{
		if(++iter>100000){
			printf("error\n"); assert(0);
		}
		key = ListDelMin(ll);

		if(key>=0){
			if(values[key]==0){
				printf("Thread %d: Error, key %d was deleted without inserting it first. \n",
						info->id, key);
				assert(0);
			}
			values[key]=0;
		}
	}while(key!=-1);

	PRINT("Thread %d has finished\n", info->id);
	return NULL;
}
/********************************************************************************************************/

/********************************************************************************************************/
// testLLDeletes - main function that dispatches the threads and tests the concurrent deletes in LL
void testLLDeletes(LinkedList *ll){

	int *values = new int[sizeForTest];
	assert( values!=NULL );
	int insCnt = 0;

	values[0]=0;						// initialize the LL sequentially, with increasing keys, and without
	for(unsigned int i=1; i<sizeForTest; ++i){	// repetitions
		int rand = 				
				abs((int)simRandom());	// get random number without lock, always returns the same numbers
		values[i]=rand%2;				// update the values array for the future validation
		if(values[i]) {
			ListInsert(ll, i);
			insCnt++;
		}
	}

	printf("LL with %d elements is ready.... GO!\n", insCnt);

	pthread_t 	threads[numOfThreads];
	ThrInf 		infos[numOfThreads];
	for(int i=0; i<numOfThreads; ++i){
		infos[i].pq=(ChunkedPriorityQueue*)ll;
		infos[i].values	= values;
		infos[i].id		= i;
		pthread_create(&threads[i], NULL, &threads_ll_del, &infos[i]);
	}

#ifndef MAC
	timespec time1, time2;				// start the time measurement
	clock_gettime(CLOCK_MONOTONIC, &time1);
#else
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
#endif

	run = true;							// let the threads go

	for(int i=0; i<numOfThreads; ++i){
		pthread_join(threads[i], NULL);
	}

#ifdef MAC
	chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();
	printf("LinkedList: time = %lld nanoseconds\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
	printf("%lld = LinkedList^: time in microseconds\n", std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
#else
	clock_gettime(CLOCK_MONOTONIC, &time2);	// end the total time timer
	printf("LinkedList: time = %ld nanoseconds\n", diff(time1,time2).tv_nsec);
	printf("%ld = LinkedList^: time in microseconds\n", (diff(time1,time2).tv_nsec / 1000));
#endif
	printf("Execution finished successfully\n");
}
/********************************************************************************************************/







/***************************************************************************************/
/********************************* MAIN FUNCTION ***************************************/
/***************************************************************************************/
/***************************************************************************************/
//ChunkedPriorityQueue _cpq;
int main(int argc, char *argv[]){

	gsl_rng *rng;

	if (argc < 5 || strcmp(argv[1]=="-h")==0) {
		printf("Please provide 4 parameters: \n1st: number of threads; 2nd: initial number of elements;"
				" 3d: 1 - insert only, 2 - delete only, 3 - mixed, 4 - mixed DES; and 4th: seconds for measurement "
				"throughput (can be float for milli/micro seconds)\n");
		exit(0);
	}

	numOfThreads = atoi(argv[1]);	// initializing the global variables
	sizeForTest  = atoi(argv[2]);
	int testType = atoi(argv[3]);
	sleepTime 	 = atof(argv[4]);

	DEB(							// single thread test for debugging only
			preliminaryTesting();
	testSingleThread();
	printf("*******  Simple testing was successful! Size for test: %d *******\n", sizeForTest);
	int values[sizeForTest];
	memset(values, 0, sizeof(int)*sizeForTest);
	printf("*******  Successful stack allocation for %d bytes! *******\n", sizeof(int)*(int)sizeForTest);
	)

	ChunkedPriorityQueue *pq = new ChunkedPriorityQueue();
	simSRandom(1000);
	printf("Stage I start: after initialization.\n");

	if (testType == 4) {
		rng = gsl_rng_alloc(gsl_rng_mt19937);
		gsl_rng_set(rng, time(NULL));
		des_workload = true;
		exps = (unsigned long *)malloc(sizeof(unsigned long) * EXPS);
		gen_exps(exps, rng, EXPS, 1000);
	}
	// STAGE I: make concurrent initial insertions as Stage I, without measuring time or throughput
#ifdef DEBUG						
	testInsertsNoMeasurements(pq, values);
#else
	testInsertsNoMeasurements(pq, NULL);
#endif	// DEBUG

	DEB(pq->compareStructure(values, sizeForTest);)
	PRINT("*******  Concurrent insertions and validation were successful!  *******\n");
	simSRandom(2000);

	// STAGE II: 
	if (testType == 1) { 			// test only insertions throughput in a second, print the results
		testInserts(pq);  
	} else if (testType == 2) {		// test only deletions throughput in a second, print the results
		testDeletesAndMixed(pq, true, NULL);	
	} else if ((testType == 3) || (testType ==4)){ 	// test mixed throughput (false in boolean parameter) in a second, 
#ifdef DEBUG						// print the results
		testDeletesAndMixed(pq, false, values);	
#else
		//		testDeletesAndMixed(pq, false, NULL);
		test80Del20Ins(pq, false, NULL);
#endif	// DEBUG
	}

	delete pq;	
}
/***************************************************************************************/



/***************************************************************************************/
/**************************** VALIDATION TESTING ***************************************/
/***************************************************************************************/
/***************************************************************************************/
int testSingleThread()
{
	ChunkedPriorityQueue *pq = new ChunkedPriorityQueue();
	int 	SIZE 		= 2000;
	int 	validationArray[(SIZE+10)];
	int 	curValid 	= 0;
	ThrInf 	info; 	info.id = 1;

	for(int i=0; i<SIZE; ++i){
		int key = rand()%200+1;
		validationArray[i] = key;
		pq->insert(key, &info);
	}

	// Turn on the printing in case some assertion fails
	cout << "<----" << SIZE << " inserts are done! Hereby printing out the resulting PQ" << endl;
	pq->print(); cout << endl << endl;

	// Compare with sorted array, to see that the values are truly ordered and none is missing
	std::sort(validationArray, validationArray+SIZE);

	int val;
	val = pq->delmin(&info);
	cout << "<----" << "First Delete Min returns: val = " << val << endl;
	pq->print(); cout << endl << endl;
	if (  val != validationArray[curValid++]  ) {
		cout 	<< "Validation error on value " << validationArray[curValid-1] << " at index " 
				<< (curValid-1) << endl;
		assert(false);
	}

	val = pq->delmin(&info);
	cout << "<----" << "Second Delete Min returns: val = " << val << ". Printing the PQ below:" << endl;
	//pq->print(); cout << endl << endl;
	if (val != validationArray[curValid++]) {
		cout << "Validation error on value " << validationArray[curValid-1] << " at index " << (curValid-1) << endl;
		assert(false);
	}

	cout << "<----" << "Inserting a key 3 that should go to buffer and cause a merge." << endl;
	pq->insert(3, &info);
	validationArray[SIZE] = 3;
	pq->print(); cout << endl << endl;

	cout << "<----" << "Inserting the last key 1 ..." << endl;
	pq->insert(1, &info);
	validationArray[SIZE+1] = 1;
	std::sort(validationArray+curValid, validationArray+SIZE+2);
	pq->print(); cout << endl;

	cout << "Validation Array from the start (starts from " << curValid << "): [0]" << validationArray[0] 
													   << ", [1]" << validationArray[1] << ", [2]" << validationArray[2] << ", [3]" << validationArray[3]
																									   << ", [4]" << validationArray[4] << ", [5]" << validationArray[5] << ", [6]" << validationArray[6] << ", [7]" << validationArray[7]
																																									    << endl;

	for(int i=0; i<SIZE; ++i){
		val = pq->delmin(&info); 
		cout<< "val = " << val << endl;
		if (val != validationArray[curValid++]) {
			cout << "Validation error on value " << validationArray[curValid-1] << " at index " 
					<< (curValid-1) << ", " << val << " is returned" << endl;
			assert(false);
		}
	}

	delete pq;
	return 0;
}
/***************************************************************************************/



/***************************************************************************************/
void preliminaryTesting() {

	// Turn on together with the printing
	DEB(Atomicable a = {{{0}}}; 					// just testing the atomic status functionality
	a.bword.idx = 1024; 
	a.bword.state = FREEZING; 
	a.bword.frozenIdx = 960;)

					_assert( sizeof(Atomicable) == 4 );			// size of status is supposed to fit into 32 bits
	_assert(pow(2.0,BITS_FOR_FR_IDX)>=MAX_IDX);	// frozen index must have enough bits to hold the maximal idx
	_assert(a.bword.idx == Atomicable::getIdx(a.iword));// same bit-field and integer representations
	_assert(CHUNK_SIZE == MAX_IDX);				// validity of max index
	_assert(a.bword.state == Atomicable::getState(a.iword));// same bit-field and integer representations

	//cout 	<< "Atomicable size is " << sizeof(Atomicable) 
	//		<< " bytes. The maximal invalid index is " << MAX_IDX << ". Chunk size: " << CHUNK_SIZE << endl;

	// Turn on the printing in case some assertion above fails
	// cout << "Atomicable size is " << sizeof(Atomicable) << " bytes. The maximal invalid index is " <<MAX_IDX;
	// cout << ". The index from bword is " << a.bword.idx << ", the index from integer is ";
	// cout << Atomicable::getIdx(a.iword)  << endl << "The state name from bword is ";
	// cout << a.printState() << ", the state number from integer is " << Atomicable::getState(a.iword);
	// cout << "The frozen index from bword is " << a.bword.frozenIdx << endl;
	// cout << "Single chunk takes " << sizeof(Chunk) << " bytes." << endl;


	return;
}
/***************************************************************************************/

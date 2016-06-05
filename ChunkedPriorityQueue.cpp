/*
 ** The code represent an efficient, concurrent and lock-free Priority Queue (PQ) with scalable deletions.
 ** The PQ's data structure is a linked list of chunks, each of which includes an array of values.
 **
 **
 ** Authors: Nachshon Cohen, Anastasia Braginsky
 */



#include <iostream>
#include <string.h>				// for memset usage
#include <algorithm>			// for sort
#include <unistd.h>

#include "rand.h"
#include "globals.h"
#include "ChunkedPriorityQueue.h"
#include "test.h"
#include "Atomicable.h"

using namespace std;



/******************************************************************************************************/
// checkInputInsert(int key)	- only keys between MIN_VAL and MAX_VAL (not included) are acceptable
static inline void checkInputInsert(int key){
	if ((key <= MIN_VAL)||(key >= MAX_VAL)){
		cout<<"NOT VALID KEY"<<endl;
		_assert(false);
	}
	//PRINT2("   <<< Key %d is going to be inserted into the PQ\n", key);
}
// end of checkInputInsert
/******************************************************************************************************/

/******************************************************************************************************/
// recursiveRecovery(...)	  -	performs all what is needed when chunk previous to currently frozen 
//								chunk is also frozen. Returns true when prev was updated and there is 
//								a need to continue and false otherwise.
inline bool ChunkedPriorityQueue::recursiveRecovery(ThrInf* t, Chunk** prev, Chunk* curr){

	Chunk* c = NULL; 			// new current and previous for further search
	Chunk* p = NULL;

	// There is a need to freeze previous chunk in order to get the local list of values
	// NOTE: we can call readFrozenChunk (for the above line)
	int svalsL[CHUNK_SIZE], lenL;	
	freezeChunk(*prev, svalsL, &lenL, FREEZE_RECOVERY, t);			

	c = getChunk(prev, &p);		// search the previous to previous chunk (in the list, not skip list)
	if (c == *prev) {			// we found the frozen prev in the list; p precedes prev
		PRINT2("	<<< Thread %d started to recover %p, but got to recover %p\n", t->id,curr,*prev);
		freezeRecovery(t, *prev, p, svalsL, lenL, FREEZE_RECOVERY);	// recursive recovery
	} else {					// prev is already not in the PQ list (but may be on the skip list)
#ifdef SKIP_LIST
		DEB2(int result =) skipListRemove(sl, (*prev)->meta.max);
#endif //SKIP_LIST
		PRINT2("	<<< Thread %d started to recover curr %p with prev %p (prev max:%d), but prev "
				"is frozen and not on the list. Result of removal previous from the skip list is: %d\n",
				t->id, curr, *prev, (*prev)->meta.max, result);
	}

	// after initial prev was recovered, re-search for initial curr and find its new predecessor
	c = getChunk(&curr, &p);
	if (!c) {  					// the frozen current chunk is already not in the list, return
#ifdef SKIP_LIST
		skipListRemove(sl, curr->meta.max);
#endif //SKIP_LIST
		return false;
	} else { *prev = p; return true; }
}
// end of recursiveRecovery
/******************************************************************************************************/


/******************************************************************************************************/
// ChunkedPriorityQueue::print() 	- prints each chunk of the priority queue
dev void ChunkedPriorityQueue::print()
{
	Chunk 	*curr 	= head;
	int 	counter	= 0;

	cout	<<	"head = "	<<	head	<<	endl;
	cout	<< 	"buffer = " << curr->meta.buffer << ". Printing buffer below if exists:" << endl;
	if (curr->clearDelBuffer()) curr->meta.buffer->print(true);

	while( curr!=NULL ){
		cout	<<	"printing chunk "	<<	++counter	<<	" ptr="	<<	curr	<<	endl;
		curr->print(true);			// printing the chunk with all the details
		curr	=	(Chunk*)curr->clearDelNext();
	}
}
// end of ChunkedPriorityQueue::print()
/******************************************************************************************************/


/******************************************************************************************************/
// ChunkedPriorityQueue::debugPrintLoopInInsert(...)	- printing for debugging issues
inline void ChunkedPriorityQueue::debugPrintLoopInInsert(	
		ThrInf* t, Chunk* prevPrev, Chunk* prevCurr, int key,
		Chunk* prev, Chunk* curr, int iter, int r){

	if (prevPrev)
		printf("   <<< Thread %d for key %d is repeatedly getting (%d times): prev=%p and curr=%p from getChunk, prevPrev = %p (%s), prevCurr=%p (%s).\n",
				t->id, key, iter, prev, curr, prevPrev, prevPrev->meta.status.printState(), prevCurr, prevCurr->meta.status.printState()  );
	else
		printf("   <<< Thread %d for key %d is repeatedly getting (%d times): prev=%p and curr=%p from getChunk, prevPrev = %p (-), prevCurr=%p (%s).\n",
				t->id, key, iter, prev, curr, prevPrev, prevCurr, prevCurr->meta.status.printState()  );

	Chunk* prevForCheck = NULL;
	Chunk* prevForSlCheck = NULL;
	Chunk* c = getChunk (&curr, &prevForCheck);

	Chunk* curD = NULL; Chunk* prevD = NULL;
	getChunkDebug(&curD, &prevD, key);

	Chunk* curSkipL = NULL;
#ifdef SKIP_LIST
	skipListContains(sl, key, (intptr_t*)&curSkipL);    // try to get the chunk through the skip-list
	Chunk* csl = getChunk(&curSkipL, &prevForSlCheck);
	printf("   <<< Thread %d: chunk %p (%s) with max %d and index %d was found in the skip list according to the key %d, "
			"its next: %p and in list:%p.\n",
			t->id, curSkipL, curSkipL->meta.status.printState(),
			curSkipL->meta.max, curSkipL->meta.status.bword.idx,
			key, curSkipL->meta.next, csl);
#endif

	if (c)
		printf("   <<< Thread %d: chunk %p (%s) with max %d and index %d was found in the linked list after %p.\n"
				"   <<< Results from walking on the list for key, cur:%p and prev:%p, skip-list result:%d, is current frozen:%d\n"
				"   <<< Results from Skip-List:%p",
				t->id, curr, curr->meta.status.printState(), curr->meta.max, curr->meta.status.bword.idx, prevForCheck,
				curD, prevD, r, curr->meta.status.isFrozen(), curSkipL);
	else
		printf("   <<< Thread %d: chunk %p (%s) with max %d and index %d was not found in the linked list. \n"
				"   <<< Results from walking on the list for key, cur:%p and prev:%p, skip-list result:%d, is current frozen:%d\n"
				"   <<< Results from Skip-List:%p",
				t->id, curr, curr->meta.status.printState(), curr->meta.max, curr->meta.status.bword.idx,
				curD, prevD, r, curr->meta.status.isFrozen(), curSkipL);

	assert(0);
}
// end of debugPrintLoopInInsert
/******************************************************************************************************/


/******************************************************************************************************/
// ChunkedPriorityQueue::insert(int key) 	
//		- insert the key into PQ, the key must be positive and require less than 32 bits
dev void ChunkedPriorityQueue::insert(int key, ThrInf* t){

	Chunk* curr = NULL, *prev = NULL;
	int iter = 0; 
	Chunk* prevCurr = NULL; Chunk* prevPrev = NULL;	// for debugging only!
	checkInputInsert(key);

#ifdef FLAG_RELEASE
	if (flag==1) count++;
	while (flag==1) { // the deletes are going
		usleep(0);
		if(count<t->num){
			count = 0;
			atomicCAS(&flag,1,0);
		}
	}
#endif //FLAG_RELEASE

	while(1) {

		int r = getChunk(&curr, &prev, key);		// set the curr to point to the relevant chunk

		if ((iter>35) && (prevCurr==curr)) {
			Chunk* curSkipL = NULL;
			Chunk* prevForSlCheck = NULL;
#ifdef SKIP_LIST
			skipListContains(sl, key, (intptr_t*)&curSkipL);
			if (!getChunk(&curSkipL, &prevForSlCheck)) {
				skipListRemove(sl, curSkipL->meta.max);
			}
#endif //SKIP_LIST
		}

		if (((iter++ > 50) && (prevCurr == curr)) || iter>1000000)
			debugPrintLoopInInsert(t, prevPrev, prevCurr, key, prev, curr, iter, r);

		prevCurr = curr; prevPrev = prev;			// for debugging only!

		if ( head == curr ){						// if this is the 1st chunk (being first is irreversible)
			_assert(curr->meta.status.isDelete() || curr->meta.status.isInFreeze());
			if(insert_to_first_chunk(t, key, curr))
				break;								// SUCCESSFUL FINISH after insertion into 1st chunk
		}

		// take care for internal chunk or frozen first chunk
		int  status = curr->meta.status.aInc();	// increase the index and get the previous status as integer
		int  idx 	= Atomicable::getIdx(status);
		bool lateFreeze = false;				// for the case where freeze happened after the first check

		if ( (idx<MAX_IDX) && (!Atomicable::isInFreeze(status)) ){
			// not full and not frozen chunk
			_assert(curr->meta.status.isInsert() || curr->meta.status.isInFreeze());
			curr->vals[idx] = key;				// use simple write for the chunk update
			membarrier();						// memory fence to make the writes globally visible
			if(!curr->meta.status.isInFreeze())
				break;							// SUCCESSFUL FINISH after verifying that chunk wasn't frozen
			else lateFreeze = true;
		}

		PRINT2(	"   <<< Thread %d inserted key %d in chunk %p at index %d (prev:%p, head:%p),"
				" when a need to freeze was found. Currently chunk in state: %s\n", 
				t->id, key, curr, idx, prev, head, curr->meta.status.printState());

		// Till here we are opportunistic (fast path), from here we start with SLOW PATH - help in freeze.
		// NOTE: For the late freeze no need to freeze the entire chunk, just help to freeze the relevant 
		// part and then check if the new key was seen in freeze. Here freezing full chunk for simplicity.
		int svals[CHUNK_SIZE], len;					 	// local variables for chunk freeze
		t->key = key;
		freezeChunk(curr, svals, &len, INSERT_FULL, t); // freeze just reads the values from frozen chunk

		if (lateFreeze)
			// the current chunk is frozen and immutable, check whether new key was seen in the freeze,
			// if yes, we are done, else, finish the freeze recovery and retry
			if ( curr->isIdxFrozen(idx) ) break;// FINISH without accomplishing the full recovery		

		PRINT(	"   <<< insert: Thread %d after freezing chunk %p (prev:%p, head:%p),"
				" got %d frozen entries. Currently chunk in state: %s\n", 
				t->id, curr, prev, head, len, curr->meta.status.printState());
		freezeRecovery(t, curr, prev, svals, len, INSERT_FULL);	// full recovery, to remove frozen chunk from PQ
		// TODO: in-freeze-help and loop
	} // end of while
	//PRINT2("       Key %d inserted into the PQ into chunk %p\n", key, curr);
	return;
}
/***************************************************************************************/



/***************************************************************************************/
// ChunkedPriorityQueue::create_insert_buffer(int key, Chunk *currhead, Chunk **curbuf) 	
// 		creates a new buffer (with the given key) assuming buffer wasn't exist; returns true if it was
// 		successful. If buffer pointer is already not NULL, because freeze has started or someone else 
// 		attached another buffer, returns false. In case freeze has started "curbuf" is going to be marked
// 		as deleted (DELETED_PTR).
bool ChunkedPriorityQueue::create_insert_buffer(int key, Chunk *currhead, Chunk **curbuf){

	Chunk *buf = alloc();

	buf->init(MIN_VAL, NULL, BUFFER, 1);// the max/next is initiated to MIN_VAL/NULL because never used
	buf->vals[0] = key;					// already created with the intended value, so index starts at 1

	if(  ( *curbuf = (Chunk*)atomicCAS((void**)&currhead->meta.buffer, NULL, buf) )   == NULL ) {
		PRINT2("The buffer %p was attached to %p\n", buf, currhead);
		*curbuf = buf;
		return true;					// *curbuf holds pointer to the buffer (ours or someone's else)
	}

	return false;
}
/***************************************************************************************/



/***************************************************************************************/
// ChunkedPriorityQueue::insert_to_first_chunk(int key, Chunk *currhead) 	
// 		- inserts the key into the first chunk (currhead is needed for possible later update due to freeze).
// 		As it is impossible to insert to the first chunk, instead the key is inserted into the buffer
bool ChunkedPriorityQueue::insert_to_first_chunk(ThrInf* t, int key, Chunk *currhead){

	Chunk *curbuf = currhead->meta.buffer;		// PHASE I: insertion into the buffer
	t->insInFirstCnt++;
	int idx = 0;

	PRINT2("Thread %d is starting insert to the first chunk %p with buffer %p\n", t->id, currhead, curbuf);

	do {										// using do-while(0) for easier skipping for PHASE II
		if( (intptr_t)(curbuf) & DELETED_PTR ) 
			return false;					// we are in process of merging the first chunk

		if( curbuf == NULL ){					// allocation of the buffer if not yet allocated
			if (create_insert_buffer(key, currhead, &curbuf)) {
				PRINT2("       Key %d inserted into the PQ into buffer %p\n", key, curbuf);
				break; 							// the key was added, go to PHASE II
			}
			else if( (intptr_t)(curbuf) & DELETED_PTR ) 
				return false;					// we are in process of merging the first chunk
		}
		_assert(curbuf->meta.status.isBuffer() || curbuf->meta.status.isInFreeze());	
		//int status  = curbuf->meta.status.aInc();	// insert into existing buffer
		int status  = curbuf->meta.status.aIncK(SKIP_ENTRIES_IN_BUFFER); // insert into existing buffer

		idx 	= Atomicable::getIdx(status);

		if ( (idx<MAX_IDX) && (!Atomicable::isInFreeze(status)) ){
			curbuf->vals[idx] = key;			// not full and not frozen chunk
			membarrier();
			if( curbuf->meta.status.isInFreeze()==false ) 
				break;							// there was no concurrent freeze, go to PHASE II
			int svals[CHUNK_SIZE], len;			// local variables for chunk freeze, here chunk is in freeze
			freezeChunk(currhead,svals,&len		// if the chunk is frozen, freezeChunk just reads the values
					,INSERT_FIRST_HEAD_HELP, t);	// for same flow first chunk is the start point of the freeze
			freezeChunk(curbuf,svals,&len,		// after first chunk is frozen freeze the buffer	
					INSERT_FIRST_BUFFER, t);
			if ( curbuf->isIdxFrozen(idx) )		// verify whether this value was taken in the freeze
				return true;					// FINISH without accomplishing the full recovery
		}				

		// the value wasn't considered, or for other reason need to recovery from the freeze and retry
		return false;
	} while(0);							

	// PHASE II: after value was inserted into the buffer, verify that first chunk is merged with the buffer
	// upon this insert end
	if(currhead->meta.status.isInFreeze()){
		// If first chunk is in freeze after we inserted the key into the buffer, this insert linearization
		// point can happen immediately now, anyway deletions can get its linearization point only after
		return true;
	}
	else{
#ifdef ELIMINATION
		//if (!(key > curbuf->vals[curbuf->meta.status.bword.idx]))
		usleep(0);                             // wait for elimination possibility
		if (curbuf->vals[idx] & ELM_TAKEN) {    // check if the key was taken
			//printf("insert to first finished without recovery\n");
			t->eleminCnt++;
			return true;
		}
#endif //ELIMINATION
		int svals[CHUNK_SIZE], len;			// local variables for chunk freeze
		// the insert need first chunk to be merged with buffer, therefore initiate freeze of the first chunk
		freezeChunk(currhead, svals, &len, INSERT_FIRST_HEAD, t);			
		freezeRecovery(t, currhead, NULL, svals, len, INSERT_FIRST_HEAD);
		return true;
	}
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::delmin() - main interface that deletes the minimal value from the PQ
// 		What does it return when the PQ is empty?
int ChunkedPriorityQueue::delmin(ThrInf* t){
	DEB(int iter=0;)

#ifdef FLAG_RELEASE
        		if (flag==0) count++;
	while (flag==0) { // the deletes are going
		usleep(0);
		if(count<t->num){
			count = 0;
			atomicCAS(&flag,0,1);
		}
	}
#endif //FLAG_RELEASE

	while(1){
		DEB(if(iter++>1000000) assert(0);)

#ifdef ELIMINATION
        		do {
        			Chunk* b = head->meta.buffer;                       // try to take one entry from the buffer
        			// int numOfTrials = 12;
        			if (b==NULL) break;
        			if ( (intptr_t)b & DELETED_PTR) break;
        			if (t->num < 2) break;
        			int limitIdx = b->meta.status.bword.idx;            // take the index of the last word already written to the buffer

        			for (int j=0; j<1; j++) {

        				int randomInteger = abs((int)simRandom()); // positive random integer
        				int bIdx = randomInteger % t->num;         // bounded by the number of threads
        				bIdx = bIdx * SKIP_ENTRIES_IN_BUFFER;      // multiplied by the empty entries

        				bIdx = limitIdx - bIdx;
        				if (bIdx < 0) bIdx = 0;

        				// for (int bIdx=0; bIdx<=limitIdx; ++bIdx) {
        				unsigned int bKey = b->vals[bIdx];                  // read a key from buffer

        				if ( bKey &&  !(bKey & ELM_TAKEN) ) {               // if the buffer key exists (not zero) and not taken (MSB not set)

        					unsigned int fKey =                         // read a key from first chunk (the rest of the keys are greater)
        							head->vals[head->meta.status.bword.idx];

        					if (bKey < fKey) {                          // if there is a smaller key in the buffer, set its MSB

        						//printf("   <<< Thread %d - key %d suitable for elemination was found at index %d\n", t->id, bKey, bIdx);

        						if (atomicCAS(&b->vals[bIdx],bKey,(bKey | ELM_TAKEN)) == (int)bKey) {
        							//printf("   <<< Thread %d - key %d was marked for elimination. Size of int: %d\n", t->id, bKey, (int)sizeof(int));
        							if (!b->isIdxFrozen(bIdx))   {      // if there was no freeze for this entry so far
        								//printf("   <<< Thread %d - key %d was eliminated\n", t->id, bKey);
        								return bKey;
        							}
        						}
        					}
        				}
        			}

        		} while (0);
#endif //ELIMINATION


		Chunk* curr = head;	Chunk* next;
		_assert(curr->meta.status.isDelete() || curr->meta.status.isInFreeze());

		int  status = curr->meta.status.aInc();             // increase the index and get the previous status as integer
		int  idx 	= Atomicable::getIdx(status);

		PRINT2("   <<< Thread %d - Delete the minimum at index %d from chunk %p,"
				" which is in state %s.", t->id, idx, curr, curr->meta.status.printState());

		if ( (idx<MAX_IDX) && (!Atomicable::isInFreeze(status)) ) {
			// deleting from not a full and not a frozen chunk
			PRINT2("The minimum is %d (Thread %d)\n", curr->vals[idx], t->id);
			return curr->vals[idx];		
		}

		next = curr->meta.next;
		_assert(next);                                      // there always have to be some next to the first
		// empty initial first chunk immediately causes its freeze, but don't freeze if there is nothing in
		// the second chunk. In this case the entire PQ is empty.
		if( !((intptr_t)next & DELETED_PTR) )               // if next ptr is not marked as deleted
			if ( (curr->meta.max==MIN_VAL) && (next->meta.max==MAX_VAL) && (next->meta.status.getIdx()==MIN_IDX) ) {
				if (idx > 50000)                            // if index is too high because PQ is empty for the long time
					curr->meta.status.aDec();               // decrease the index
				return -1;
			}

		int svals[CHUNK_SIZE], len;                         // local variables for chunk freeze
		freezeChunk(curr, svals, &len, DELETE_MIN, t);      // freeze just reads the values from frozen chunk
		freezeRecovery(t, curr, NULL, svals, len, DELETE_MIN);	// the recovery
	}
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::readAndFreezeValues(Chunk *curr, int *sortedval, int *len, bool isFirstChunk)
// 		Freezes the values inside the Chunk curr, return locally.

void ChunkedPriorityQueue::readAndFreezeValues(Chunk *curr, int frozenIdx, int *sortedval, int *len, bool isFirstChunk){

	if( frozenIdx < CHUNK_SIZE ){                       // frozen index should always be less then a chunk size

		TODO("beautify the code below. Should be in a different function")

        		int j=0, prevj=0;

		for(int k=0; k<CACHE_LINES_IN_CHUNK-1; ++k){    // go over parts of the chunk's values that can be
			// indicated in one exist-entry
			unsigned exist=0, mask=1;

			for(int i=frozenIdx; i<VALS_PER_EXIST; ++i, mask<<=1){
				// prepare a mask for one exist-entry, but what if frozenIdx is initially much higher than VALS_PER_EXIST?
				int val = curr->vals[i+k*VALS_PER_EXIST];
#ifdef ELIMINATION
				if ((curr->meta.status.getState()==BUFFER) && (val & ELM_TAKEN)) continue;   // in the first chunk the value  taken
				if ((curr->meta.status.getState()==BUFFER) && (i%SKIP_ENTRIES_IN_BUFFER==0)) continue;
#endif      //ELIMINATION
				if(val){
					exist|=mask;
					sortedval[j++]=val;
				}
			}

#ifndef FIRST_CHUNK_EXPLICIT_FREEZE
			if (!isFirstChunk) {
#endif //FIRST_CHUNK_EXPLICIT_FREEZE
				u64 existl = DIRTY_EXIST | (u64)exist;	// freeze only one word
				u64 existres = atomicCAS(&curr->meta.exist[k], 0ull, existl);

				if( existres == 0 || existres == existl){
					// existres==0: we successfully set exists.
					// existres==existl: someone else set exist, but it set to exactly the same value we gathered. No need to re-read anything.
					; // DO NOTHING
				} else { // someone else set exists entry, and disagree with us. Re-read anything.
					j=prevj;
					exist=(unsigned)existres;//trim the dirty bit. retain the first 32 bits.
					for(int i=frozenIdx; i<VALS_PER_EXIST && exist; ++i){
						if(exist&1){
							sortedval[j++] = curr->vals[i+k*VALS_PER_EXIST];
							_assert(curr->vals[i+k*VALS_PER_EXIST]!=0);
						}
						exist>>=1;//now lsb is the i-th bit
					}
				}
#ifndef FIRST_CHUNK_EXPLICIT_FREEZE
			}
#endif //FIRST_CHUNK_EXPLICIT_FREEZE

			prevj=j;
			frozenIdx-=VALS_PER_EXIST;
			frozenIdx=(frozenIdx<0)?0:frozenIdx;
		}

		*len = j;
		std::sort(sortedval, sortedval+j);
	}
#ifdef DEBUG
	else{
		for(int k=0; k<CACHE_LINES_IN_CHUNK-1; ++k)
			curr->meta.exist[k]=DIRTY_EXIST;
	}
#endif

}


/******************************************************************************************************/
// ChunkedPriorityQueue::freezeChunk(Chunk *curr, int *sortedval, int *len) 
// 		- Freezes the chunk pointed by curr and creates an sorted array (sortedval) out of frozen values.
// 		The length of array is - len
void ChunkedPriorityQueue::freezeChunk(Chunk *curr, int *sortedval, int *len, FreezeTrigger ft, ThrInf* t){

	int idx;
	Atomicable ns, status;                          // new status, current status
	int frozenIdx = 0;
	*len = 0;                                       // initialize the input-output parameter
	bool isFirstChunk = false;
	DEB(int iter = 0;)

	while(1){                                       // PHASE I : set the relevant chunk status if needed
		DEB(if(iter++>1000000) assert(0);)
        		status = curr->meta.status;                 // read the status and get its state
		ns = status;                                // new status
		idx = status.getIdx();

		switch (status.getState()){
		case BUFFER	:                           // always successful freeze, frozenIdx are not used
		case INSERT :                           // in insert or buffer chunks
			// NOTE: it is important that FREEZING = 011, FROZEN=111, and other states = 0xx (binary)
			curr->meta.status.aOr(FREEZING<<BITS_FOR_IDX);
			break;

		case DELETE	:
			isFirstChunk = true;
			if (idx >= MAX_IDX) {
				ns.set(FREEZING, idx, CHUNK_SIZE);
				frozenIdx = CHUNK_SIZE;
			} else {
				ns.set(FREEZING, idx, idx);
				frozenIdx = idx;
			}
			_assert(ns.isInFreeze());           // try to CAS to the new state, in the next line
			if ( curr->meta.status.bCAS(status.iword, ns.iword) ) {
				break;
			}
			else { continue; }                  // CAS can be avoided by a delete

		case FREEZING:
			frozenIdx = status.bword.frozenIdx;
			break;

		case FROZEN	:                           // someone else has frozen this

			if (idx > 20000) {
				PRINT("   <<< Thread %d - Freezing frozen %p, trigger: %d. The index and frozen idx "
						"are %d and %d. Iterator: %d. Head: %p. Key: %d\n", 
						t->id, curr, ft, status.bword.idx, status.bword.frozenIdx, iter, head, t->key);
			}
			readFrozenChunk(curr, sortedval, len);
			curr->markPtrs();                   // set next chunk pointer as deleted
			return;                             // the chunk is frozen, there is nothing to do

		default: _assert(false);
		}	
		// break the loop reaching this line, continue only if CAS from DELETED status failed
		break;						
	}
	_assert(curr->meta.status.isInFreeze());

	// PHASE II: freeze the entries and create the local array of frozen values
	readAndFreezeValues(curr,                       // entries of the first chunk are not frozen, because it is not needed
			frozenIdx, sortedval, len, isFirstChunk);

	PRINT2("   <<< freezeChunk: Thread %d - After freezing chunk %p which is now in state %s. The previous state is %s and the length of the values array is %d\n", 
			t->id, curr, curr->meta.status.printState(), status.printState(), *len);

#ifdef DEBUG
	readFrozenChunk(curr, sortedval, len); //make sure that anything is ok
#endif

	PRINT2("   <<< freezeChunk: Thread %d - After rechecking chunk %p which is now in state %s. The previous state is %s and the length of the values array is %d\n", 
			t->id, curr, curr->meta.status.printState(), status.printState(), *len);

	curr->markPtrs();										// set the chunk pointers as deleted
	TODO("Do not mark the buffer ptr unless in first chunk")
	curr->meta.status.aOr(FROZEN<<BITS_FOR_IDX); //we move from state 3 (freezing) to state 7 (frozen) using Or instead of CAS loop.
	_assert(curr->meta.status.isFrozen());
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::readFrozenChunk(Chunk *curr, int sortedval[], int *len) 
// 		Creates an array of sorted values while the chunk (and all its entries) are already frozen.
void ChunkedPriorityQueue::readFrozenChunk(Chunk *curr, int sortedval[], int *len){

	_assert(curr->meta.status.isInFreeze());

	int frozenIdx = curr->meta.status.bword.frozenIdx;

	if(frozenIdx<CHUNK_SIZE){
		_assert(curr->meta.exist[0]!=0); 	//also must be true for all exists entries
	}
	else{
		*len=0; return;
	}

	int j=0;
	for(int k=0; k<CACHE_LINES_IN_CHUNK-1; ++k){
		unsigned exist=0;
		exist=(unsigned)curr->meta.exist[k];
		if(exist==SET_WORD){
			for(int i=frozenIdx; i<VALS_PER_EXIST; ++i) {
				sortedval[j++]=curr->vals[i+k*VALS_PER_EXIST];
				_assert(curr->vals[i+k*VALS_PER_EXIST]!=0);
			}
		}
		else{
			for(int i=frozenIdx; i<VALS_PER_EXIST && exist; ++i){
				if(exist&1){
					sortedval[j++] = curr->vals[i+k*VALS_PER_EXIST];
					if ( curr->vals[i+k*VALS_PER_EXIST] == 0 ) {
						PRINT("An error that the value is zero, but frozen? For i=%d and k=%d\n",i,k);
						_assert(curr->vals[i+k*VALS_PER_EXIST]!=0);
					}
				}
				exist>>=1;//now lsb is the i-th bit
			}
		}
		if(frozenIdx!=0){
			frozenIdx-=VALS_PER_EXIST;
			if(frozenIdx<0) frozenIdx=0;
		}
	}
	*len = j;
	std::sort(sortedval, sortedval+j);

#ifdef DEBUG
	for(int i=0; i<((*len)-1); ++i){
		//  _assert(sortedval[i]<=sortedval[i+1]);
		_assert(sortedval[i]!=0);
	}
	_assert(*len==0 || sortedval[*len-1]!=0);
#endif

}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::merge(Chunk *next, int svals[], int len)
//		
// - takes two frozen chunks (their sorted values) and creates the first chunk, and an the second chunk
Chunk* ChunkedPriorityQueue::merge(Chunk *next, int svals[], int len, Chunk** nextCreated){

	Chunk *second = alloc(), *first=alloc();			// local pointers to first and second chunks
	int i = 0, o = 0;

	if( len <= CHUNK_SIZE ){
		o = CHUNK_SIZE-len;								// for testing small values, put all values in the
		if (next==NULL) {
			first->init(svals[len-1], second, DELETE, o);	// first chunk, start deleting from middle	
			second->init(MAX_VAL, next, INSERT, MIN_IDX);	// second chunk is empty in this case
			*nextCreated = second;
		} else {
			first->init(svals[len-1], next, DELETE, o);
		}
		while (i<len) {first->vals[o++]=svals[i++];}
		return first;
	}

	// first and second are going to have values
	first->init(svals[CHUNK_SIZE-1], second, DELETE, MIN_IDX);	
	second->init(svals[len-1], next, INSERT, len-CHUNK_SIZE); //second chunk contains len-CHUNK_SIZE values. insert start afterward.
	*nextCreated = second;
	while (i<CHUNK_SIZE) {first->vals[o++]=svals[i++];}
	o = 0;
	while (i<len) {second->vals[o++]=svals[i++];}
	return first;
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::freezeRecovery(ThrInf* t, Chunk *curr, Chunk *prev, int svals[], int len)
// 		- recovers the PQ from the frozen chunk curr, where curr's previous chunk is prev
// 		for the first current chunk the prev is a NULL pointer
// @param curr : the current chunk to freeze
// @param prev : previous chunk, or NULL if curr is the first chunk
// @param svals, len: the array of the sorted values inside curr, and its length
void ChunkedPriorityQueue::freezeRecovery(	ThrInf* t, Chunk *curr, Chunk *prev, int svals[], int len,
		FreezeTrigger ft){

	bool toSplit = true; 				// true if toSplit was decided or false if merge-first was decided
	Chunk* local = NULL;				// to hold the pointer to the locally created recovery structure
	bool success = false;
	Chunk *nextFrozen = NULL, *nextNextFrozen = NULL, *nextCreated = NULL;
	int i = 0;

	PRINT2("   <<< freezeRecovery: Thread %d starts freeze recovery with prev:%p and curr:%p (curr idx:%d); trigger: %d\n", 
			t->id, prev, curr, curr->meta.status.bword.idx, ft);

	while(1) {
		DEB(if (i++>1000000) assert(0);)
		// PHASE I: if previous chunk is frozen, internal chunk, its recovery needs to be finished first
		if ( (prev!=NULL) && (prev != head) && prev->meta.status.isFrozen() ) {
			if ( recursiveRecovery(t, &prev, curr) )
				continue;
			else
				return;
		}

		// PHASE II: decide how to recover from the freeze: split or merge-first
		// When to merge? If this is a call on the first chunk OR if this is the call on the second chunk,
		// but previous first chunk is in process of freezing
		if ( prev==NULL || ((prev == head) && prev->meta.status.isInFreeze()) )	toSplit = false;

		if (prev) {
			PRINT2("   <<< freezeRecovery: Thread %d, Freeze recovery for c:%p with max-key %d. Split? %d"
					", with %d frozen entries. Trigger: %d. Prev chunk: %p in state %s. Head:%p\n",
					t->id, curr, curr->meta.max, toSplit, len, ft, prev, prev->meta.status.printState(), head);
		} else {
			PRINT2("   <<< Thread %d, Freeze recovery for chunk %p with maximal key %d. Is it split? %d We"
					" have %d frozen entries. Trigger: %d. Prev chunk: %p\n",
					t->id, curr, curr->meta.max, toSplit, len, ft, prev);
		}

		// PHASE III: apply the decision locally
		if (toSplit) local = split(curr, prev, svals, len, &nextCreated);
		else	local = recoverFirstChunk(curr, svals, len, &nextFrozen, &nextNextFrozen, &nextCreated, t);

		// PHASE IV: change the PQ accordingly to the decision if decision was wrong - repeat
		if (toSplit) {
			if ( prev->nextCAS(curr, local) ) 		{		
				PRINT2("	<<< Thread %d recovered split chunk %p (max key: %d)\n", 
						t->id, curr, curr->meta.max);
				success = true;
			}
		} else {
			_assert(local->meta.status.isDelete());
			// if (prev == NULL) // Modify head, save previous head value (current chunk value) in prev
			// prev = head;
			// if (   prev == (Chunk*)atomicCAS((void**)&head, curr, local)   ) 
			// success = true;

			if(prev==NULL) prev=curr; //We modify the head chunk. So previous value of head is the current chunk value. 
			if (   prev == ((Chunk*)atomicCAS((void**)&head, prev, local))   ) {
				curr = prev;
				success = true;
			}

		}

#ifdef SKIP_LIST
		if (success) {								 
			// PHASE V: In case skip list needs to be updated
			PRINT2("	<<< Thread %d updated the next of prev:%p (head:%p) with %p, key:%d, (and next %p)"
					" after cur:%p was frozen. Split?:%d Previous chunk or previous head state: %s, was it in"
					" freeze?:%d\n",
					t->id, prev, head, local, local->meta.max, nextCreated, curr, toSplit,
					prev->meta.status.printState(), prev->meta.status.isInFreeze());

			// new two chunks were added due to split or merge of the old chunk(s). The thread who had done
			// the connection is responsible to update the skip-list: remove the pointer to the one (two or
			// three) old chunk(s) and add pointers to the new chunks
			if (!skipListRemove(sl, curr->meta.max))
				PRINT2("	<<< Thread %d didn't remove first chunk %p from skip list with key %d\n",
						t->id, curr,curr->meta.max);
			else PRINT2("	<<< Thread %d did remove first chunk %p from skip list with key %d\n",
					t->id, curr,curr->meta.max);

			// The skip-list is not precise, therefore we allow it not to be able to delete/insert something
			// sometimes. So no sanity checks for returning a successful values
			if (nextFrozen) {			
				if (!skipListRemove( sl, nextFrozen->meta.max))
					PRINT2("	<<< Thread %d didn't remove 2nd chunk %p from skip list (key:%d), after "
							"first curr %p\n", t->id, nextFrozen,nextFrozen->meta.max, curr);
				else PRINT2("	<<< Thread %d did remove 2nd chunk %p from skip list (key:%d), after first "
						"curr %p\n", t->id, nextFrozen,nextFrozen->meta.max, curr);
			}

			if (nextNextFrozen) {						
				if (!skipListRemove( sl, nextNextFrozen->meta.max))
					PRINT2("	<<< Thread %d didn't remove third chunk %p from skip list (key:%d), after "
							"first curr %p\n", t->id, nextNextFrozen,nextNextFrozen->meta.max, curr);
				else PRINT2("	<<< Thread %d did remove third chunk %p from skip list (key:%d), after first"
						" curr %p\n", t->id, nextNextFrozen,nextNextFrozen->meta.max, curr);
			}

			// Note that skip list addition can be unsuccessful because two chunks can have identical keys
			DEB2 (int res1 = 2, res2 = 2;)
			DEB2(res1 =) skipListAdd(sl, local->meta.max, (intptr_t)local);				
			if (nextCreated) DEB2(res2 =) skipListAdd(sl, nextCreated->meta.max, (intptr_t)nextCreated); 

			PRINT2("	<<< Thread %d tried to add chunks %p (k:%d, res:%d) and %p (res:%d) to skip list\n", 
					t->id, local, local->meta.max, res1, nextCreated, res2);
		} // end of if we succeeded to recover
#endif //SKIP_LIST

		return; //The thread should try again to apply its operation. (!!!!)
		// the decision was wrong, retry

	} // end of while
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::split(Chunk *curr, Chunk *prev, int svals[], int len)
// 	- create as much as needed new linked chunks from the given values
Chunk * ChunkedPriorityQueue::split(Chunk *curr, Chunk *prev, int svals[], int len, Chunk** nextCreated){

	_assert( len >= 2 );
	Chunk *pc = alloc(), *nc = NULL, *res = pc;		// allocate prev & next chunks of the local linked list

	for (int i=1; i<SPLIT_CHUNKS_NUM; ++i){
		int piv = svals[(len/SPLIT_CHUNKS_NUM)*i-1];// pick pivot, svals is ordered
		_assert(piv>0);
		nc = alloc();								// allocate next chunk
		if (i == 1) *nextCreated = nc;
		pc->init(piv, nc, INSERT, (len/SPLIT_CHUNKS_NUM));// make the previous chunk to point on the next

		fill_chunk(svals, len, 						// filling the previous chunks
				(len/SPLIT_CHUNKS_NUM)*(i-1), (len/SPLIT_CHUNKS_NUM)*i, pc);
		pc = nc;									// prepare for next iteration
	}

	nc->init(curr->meta.max, 						// take the maximal value for new chunk from old chunk
			curr->clearDelNext(), 					// clear the delete bit from the next pointer
			INSERT, len-(len/SPLIT_CHUNKS_NUM)*(SPLIT_CHUNKS_NUM-1));

	fill_chunk(svals, len, 							// filling the last chunks
			(len/SPLIT_CHUNKS_NUM)*(SPLIT_CHUNKS_NUM-1), len, nc);


	// Sanity check that all that appears in the frozen chunk appear also in the local
	DEB(	
		int cnt = 1; int prevKey = svals[0];
		for (int i=1; i<len; ++i) {
			if (svals[i] == prevKey) { 				// svals[i] appeared on some frozen chunk
				cnt++;
			} else {
				int newCnt = res->countForDebug(prevKey) +
					res->meta.next->countForDebug(prevKey);
				_assert(cnt==newCnt);
				cnt = 1;
			}
			prevKey = svals[i];
		}
	)


	return res;
}
/***************************************************************************************/


/***************************************************************************************/
// ChunkedPriorityQueue::fill_chunk(int *svals, int len, int start, int pivotIdx, Chunk *c)
// 	- fills the values of chunk c from index "start" till index "pivotIdx" from array svals
void ChunkedPriorityQueue::fill_chunk(int *svals, int len, int start, int pivotIdx, Chunk *c){

	if( pivotIdx>len ) pivotIdx = len;
	int j=0;
	for(int i=start; i<pivotIdx; ++i){
		c->vals[j++]=svals[i];
	}
}
/***************************************************************************************/



/***************************************************************************************/
// ChunkedPriorityQueue::recoverFirstChunk(Chunk *curr, int svals[], int len) - recovers 
// 	the situation where first chunk is frozen
Chunk* ChunkedPriorityQueue::recoverFirstChunk(	Chunk *curr,        // can be the head or next to head
		int svals[], int len, Chunk** nextFrozen,
		Chunk** nextNextFrozen, Chunk** nextCreated, ThrInf* t){

	int svals1[CHUNK_SIZE], svals2[CHUNK_SIZE], len1 = 0, len2 = 0;
	Chunk* next = NULL;                                             // first not frozen chunk after the recovery
	*nextFrozen = NULL; *nextNextFrozen=NULL; *nextCreated=NULL;	// initialize the input-output parameters

	// freeze buffer chunk, if already frozen, freeze will give us the values in that chunk, sorted
	if(curr->clearDelBuffer() != NULL)
		freezeChunk(curr->clearDelBuffer(), svals1, &len1, FIRST_RECOVERY_BUFFER, t);

	if(len==0 && curr->clearDelNext()!=NULL){                       // first chunk is empty, freeze second chunk if exists
		freezeChunk(curr->clearDelNext(), svals2, &len2,FIRST_RECOVERY_MORE, t);
		*nextFrozen = curr->clearDelNext();                         // update the next frozen chunk after the current
		next = (curr->clearDelNext())->clearDelNext();              // first not frozen is 3rd chunk
		if(next!=NULL && (len1+len2<(CHUNK_SIZE/2))){
			// buffer and second are not enough for half-full first chunk, freeze third if exists
			freezeChunk(next, svals, &len,FIRST_RECOVERY_MORE, t);
			*nextNextFrozen = next;
			PRINT2("   <<< When recovering first chunk %p, third chunk %p was frozen!\n", curr, next);
			next = next->clearDelNext();                            // first not frozen is 4th chunk
		}
	} else {                                                        // first chunk is not empty or there is no second chunk
		// Entries taken only from first chunk and buffer. First not frozen is 2nd chunk (next can be set to NULL).
		next = curr->clearDelNext();
	}

	PRINT2("  <<< Recovering 1st chunk %p, taking %d from first, %d from buffer %p, and %d from second %p\n",
			curr, len, len1, curr->clearDelBuffer(), len2, curr->clearDelNext());

	// take care for a special case when there are no more values remaining in the PQ
	if (len+len1+len2 == 0) {
		if (!next) {    // should happen only once, when queue becomes empty
			Chunk *second = alloc(), *first=alloc();		
			first->init(MIN_VAL, second, DELETE, CHUNK_SIZE);
			second->init(MAX_VAL, NULL, INSERT, MIN_IDX);
			printf("  <<< Queue became empty, new 1st chunk %p, new second: %p\n", first, second);
			return first;
		} else {
			printf("  <<< Recovering 1st chunk %p, taking %d from first, %d from buffer %p, and %d from second %p\n",
					curr, len, len1, curr->clearDelBuffer(), len2, curr->clearDelNext());
			cout << "NOT EXPECTEDLY 3 CHUNKS EMPTY!" << endl;
			assert(false);
		}
	}

	// now combine all frozen values together
	int alllen = len+len1+len2;
#ifndef DEBUG
	int allvals[alllen];
#else
	int allvals[CHUNK_SIZE*3];
#endif

	int intvals[len+len1];
	std::merge(svals, svals+len, svals1, svals1+len1, intvals);
	std::merge(svals2, svals2+len2, intvals, intvals+len+len1, allvals);

#ifdef DEBUG
	for(int i=0; i<alllen-1; ++i){
		_assert(allvals[i]<=allvals[i+1]);
		_assert(allvals[i]!=0);
	}
	_assert(allvals[alllen-1]!=0);
#endif

	Chunk* result = merge(next, allvals, alllen, nextCreated);

	// Sanity check that all that appears in the frozen chunk appear also in the local
	DEB(	
		int cnt = 1; int prevKey = allvals[0];
		for (int i=1; i<alllen; ++i) {
			// allvals[i] appeared on some frozen chunk
			if (allvals[i] == prevKey) {
				cnt++;
			} else {
				int newCnt = result->countForDebug(prevKey) +
						result->meta.next->countForDebug(prevKey);
				if (cnt!=newCnt) {
					PRINT(	"allvals: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,"
							" %d, %d, %d, %d, %d\n", allvals[0], allvals[1], allvals[2], allvals[3], allvals[4],
							allvals[5], allvals[6], allvals[7], allvals[8], allvals[9], allvals[10],
							allvals[11], allvals[12], allvals[13], allvals[14], allvals[15], allvals[16],
							allvals[17], allvals[18], allvals[19], allvals[20], allvals[21]);
					PRINT(	"(On iteration %d) Key %d appeared frozen %d times and copied %d times (found on first chunk %d times)\n",
							i, prevKey, cnt, newCnt, result->countForDebug(prevKey));
					result->print(true); PRINT("\n");
				}
				_assert(cnt==newCnt);
				cnt = 1;
			}
			prevKey = allvals[i];
		}
	)

	return result;
}
/***************************************************************************************/

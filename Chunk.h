/*
 ** The file gives the declaration of Chunk class
 **
 ** Authors: Nachshon Cohen, Anastasia Braginsky
 */

#ifndef CHUNK_H
#define CHUNK_H

#include <iostream>
#include <stdint.h>

#include "debug.h"
#include "globals.h"
#include "Atomicable.h"

using namespace std;


/************************************************************************************************/
/* ------------------------ CONSTANTS DEFINITIONS  ------------------------ */
const int MIN_IDX 		= 0; 							// minimal VALID value for the index in the chunk
const int MAX_IDX		= (CHUNK_SIZE);					// maximal INVALID value for the index in the chunk
const int MIN_VAL		= 0;							// minimal value to never be inserted to the PQ
const int MAX_VAL 		= (0x7fffffff-1);				// maximal value to never be inserted to the PQ, -1
// is needed to distinguish it from SL max sentinel
const int DELETED_PTR	= 0x00000001;					// the mask for the least significant bit being set
/************************************************************************************************/


/************************************************************************************************/
/* ------ CLASS: Chunk defines the single chunk consisted of array of values, ------ */
/* ------ status, pointers to other chunks and some metadata                  ------ */
class Chunk
{
public:
	struct metadata{				// the chunks meta data, takes one cache line
		Atomicable 		status;		// the index for insertion or deletion, state and frozen index
		volatile int 	max;		// constant value initialized on chunk creation
		volatile Chunk* next;		// pointer to the next chunk in the linked list
		volatile Chunk* buffer;		// used only for first chunk
		volatile u64 				// used for freeze optimization: a bit is set for each frozen value
		exist[CACHE_LINES_IN_CHUNK-1];
	} meta __attribute__ ((aligned (128)));

	int vals[CHUNK_SIZE];			// the array of values

	//---------- Chunk object methods ----------
	dev void print(bool verbose){
		cout	<< 	"idx="		<<	meta.status.getIdx()	<<	" state="	<<	meta.status.printState()
						<<	" max="		<<	meta.max	<< " next="	<<	meta.next 	<< 	" buffer=" << meta.buffer;
		cout	<<	" exist[0]="<<	hex			<<	meta.exist[0]	<<	dec	<<	endl;
		if(verbose){
			for(int i=MIN_IDX; i<CHUNK_SIZE; ++i)
				cout<<vals[i]<<" ";
			cout<<endl;
		}
	}

	// special initialization method and not a constructor is used, as chunks are statically allocated
	dev void init(int max, Chunk *next, ChunkState s, int initIdx){
		//memset(this, 0, sizeof(Chunk));
		meta.max = max;
		meta.next = next;
		_assert(meta.exist[0]==0 && meta.buffer==NULL);	// assert everything is properly zeroed by the memset
		meta.status.init(s, initIdx);					// set the initials: state and the value of the index
	}
	dev bool isIdxFrozen(int idx) {						// checks whether a value in given index was frozen
		long exist = meta.exist[idx/32]; 				// 32 is the number of vals per entry in exist array
		_assert((exist>>32)!=0);
		if((((int)exist)&(1<<(idx&0x1F))))
			return true;								// found frozen
		else return false;
	}
	dev void markPtrs() {								// marks the next pointer as deleted so none can connect 
		atomicOr((int*)&(meta.next), DELETED_PTR);		// itself to the frozen node after the node is frozen
		atomicOr((int*)&(meta.buffer), DELETED_PTR);
	}
	dev Chunk* clearDelNext() {							// returns the next pointer not marked as deleted,
		intptr_t tmp = (intptr_t)(meta.next);			// regardless whether initially it was marked as
		tmp = (tmp & ~(DELETED_PTR) );					// deleted or not
		return (Chunk*)tmp;
	}
	dev Chunk* clearDelBuffer() {
		intptr_t tmp = (intptr_t)(meta.buffer);
		tmp = (tmp & ~(DELETED_PTR) );
		return (Chunk*)tmp;
	}

	dev bool nextCAS(Chunk* expectedNext, Chunk* newNext) {
		return ( expectedNext == (Chunk*)atomicCAS((void**)&this->meta.next, expectedNext, newNext) );
	}

	int countForDebug( int key ) {
		int res = 0;
		for(int i=MIN_IDX; i<CHUNK_SIZE; ++i) {
			if (vals[i] == key) res++;
		}
		return res;
	}
};
/************************************************************************************************/

#endif // CHUNK_H

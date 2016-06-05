/*
 ** The file gives the declaration of Atomicable class
 ** this class is used to represent the status of the chunk
 **
 ** Authors: Nachshon Cohen, Anastasia Braginsky
 */

#ifndef ATOMICABLE_H
#define ATOMICABLE_H

#include "globals.h"		// for CHUNK_SIZE and atomic functionality
#include "debug.h"			// for _assert
#include <string.h> 		// for memset 

const int BITS_FOR_IDX		= 18;                       // bits' partition for the chunk's status, all together take 32 bits
const int BITS_FOR_STATE	= 3;
const int BITS_FOR_FR_IDX	= 11;                       // the chunk's valid maximal index can be only 2048 (!)

const int MASK_FOR_STATE    = ((1<<BITS_FOR_STATE)-1);  // BITS_FOR_STATE least significant bits sets to 1, rest 0
const int MASK_FOR_IDX      = ((1<<BITS_FOR_IDX)-1);    // BITS_FOR_IDX least significant bits sets to 1, rest 0



/************************************************************************************************/
/* ------ CHUNK STATE DEFINITIONS ------ */
// NOTE: it is important that FREEZING = 011, FROZEN=111, and other states = 0xx (binary)
typedef enum ChunkState_t 		// states of a chunk
{								
	INSERT = 0,		// chunk that is meant only for inserts is created in the INSERT state (000)
	DELETE, 		// chunk that is meant only for deletions is created in the DELETE state (001)
	BUFFER, 		// chunk that is meant to serve as a buffer is created in the BUFFER state (010)
	FREEZING, 		// when chunk is full, empty, or going to be immutable, it inserts FREEZING state (011)
	FROZEN = 7		// after freeze process is done the state of a chunk is changed from FREEZING to FROZEN
} ChunkState;

// forward declaration of an array for easier printing out state names
extern const char *stateNames[]; 
/************************************************************************************************/


/************************************************************************************************/
/* ------ CLASS: "Atomicable" defines the status of a chunk that can be interpreted  ------ */
/* ------ as a bit-field or as an integer. It can be only atomically updated.        ------ */
class Atomicable{

public:
	union{										// anonymous union is used to allow access to the same member
		struct{									// as bit field and as an integer
			unsigned int idx:	BITS_FOR_IDX;	// idx must be located in the least significant bits
			ChunkState state:	BITS_FOR_STATE;
			unsigned int frozenIdx:	BITS_FOR_FR_IDX;
		} bword;								// bit-fields representation
		int iword;						// integer representation
	};

	// static methods of Atomicable class are used in order to manage an integer that is locally 
	// representing the Atomicable object
	static int getIdx(int w){ 			return (w & MASK_FOR_IDX);}
	static ChunkState getState(int w){	return (ChunkState)( (w >> BITS_FOR_IDX) & MASK_FOR_STATE );}

	static bool isInFreeze(int w){				// the chunk is considered freezing if it is in FREEZING or
		return ( ((int)getState(w)>=FREEZING)	// FROZEN state, OR if its index is not relevant and thus is
				|| (getIdx(w)>CHUNK_SIZE) );		// higher than the CHUNK_SIZE. Strictly higher in order not
	}											// to consider empty first chunk as frozen.

	static bool isFrozen(int w){    	return (getState(w)==FROZEN);  	}
	static bool isDelete(int w){    	return (getState(w)==DELETE);	}
	static bool isInsert(int w){    	return (getState(w)==INSERT);	}
	static bool isBuffer(int w){    	return (getState(w)==BUFFER);	}

	// Atomicable object methods
	int getIdx(){			return (int)bword.idx;}
	ChunkState getState(){	return bword.state;}
	bool isFrozen(){		return ( bword.state == FROZEN );	}
	bool isInFreeze(){    	return isInFreeze(iword); } 	// special check
	bool isDelete(){		return (bword.state == DELETE);}
	bool isInsert(){		return (bword.state == INSERT);}
	bool isBuffer(){		return (bword.state == BUFFER);}
	const char* printState(){		return stateNames[bword.state];}

	int aInc(){									// increase the index
		int res = atomicInc(&iword, 1);
		_assert( getIdx(res)<1000000 );			// panic when the index is unreasonably high
		return res;
	}

	int aIncK(int k){							// increase the index by k
		int res = atomicInc(&iword, k);
		_assert( getIdx(res)<1000000 );			// panic when the index is unreasonably high
		return res;
	}

	int aDec(){									
		_assert( bword.idx > 10000 );			// panic when the index is unreasonably low
		int res = atomicDec(&iword, 100);		// decrease the index

		return res;
	}

	int aOr(int mask){							// set the bits from the mask
		int res = atomicOr(&iword, mask); 
		_assert( getIdx(res|mask)<10000 ); 
		return res; 
	}

	void init(ChunkState s, int idx){			// initiate a new Atomicable object
		memset(this, 0, sizeof(Atomicable));
		bword.idx = idx;						// initiate the Atomicable fields: state and idx,
		bword.state = s;						// frozen index is never set at the initiation
		_assert(getIdx(iword)<10000);			// panic when the index is unreasonably high
	}
	void set(ChunkState s, int idx, int frIdx);	// set an Atomicable object to a specific value
	bool bCAS(int expected, int val);			// used in order to change the status of a chunk, returns
	// true if the atomic operation was successful
	int CAS(int expected, int val);	 			// used in order to change the status of a chunk, returns
	// the previous value of an object as integer
};
/************************************************************************************************/

#endif  // ATOMICABLE_H

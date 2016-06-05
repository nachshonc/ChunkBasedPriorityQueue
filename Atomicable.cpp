/*
 ** The file gives some implementation of Atomicable class
 **
 ** Authors: Nachshon Cohen, Anastasia Braginsky
 */

#include "Atomicable.h"

const char *stateNames[] = { 	
		const_cast<char *>("INSERT"),		//0 	000
		const_cast<char *>("DELETE"),		//1 	001
		const_cast<char *>("BUFFER"),		//2 	010
		const_cast<char *>("FREEZING"),		//3 	011
		const_cast<char *>("FROZEN"),		//4 	100
		const_cast<char *>("FROZEN"),		//5 	101
		const_cast<char *>("FROZEN"),		//6		110
		const_cast<char *>("FROZEN") };		//7		111



/************************ CLASS METHODS IMPLEMENTATIONS ******************************************/
bool Atomicable::bCAS(int expected, int val){			
	_assert(getIdx(val)<10000);				// returns true if the atomic operation was successful
	return	__sync_bool_compare_and_swap(&(iword), expected, val);
}

int Atomicable::CAS(int expected, int val){	// used in order to change the status of a chunk
	_assert(getIdx(val)<10000);				// returns the previous value of Atomicable object as integer
	return atomicCAS(&iword, expected, val);    
}

void Atomicable::set(ChunkState s, int idx, int frIdx){	
	bword.frozenIdx = frIdx;			// used to set an Atomicable object to a specific value
	bword.idx = idx;
	bword.state = s;
	_assert(s!=BUFFER);					// panic when this method is used to set the BUFFER state
	_assert(getIdx(iword)<10000);		// panic when the index is unreasonably high
}
/************************************************************************************************/

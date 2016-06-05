#ifndef GLOBALS_H
#define GLOBALS_H

typedef unsigned long long u64;

#define volatile						//use explicit atomic instructions instead of volatile!
/*******************************************************************************************************/

/****************************************** MACHINE 32/64 **********************************************/
//#define FREEZE_64						// enable to freeze using CAS on 64 bits word, if not defined 32 bits
// words are used
/*******************************************************************************************************/

/************************************* Chunk related definitions *****************************************/
//#define CACHE_LINES_IN_CHUNK 	31		// number of cache lines used for arrays of values in a chunk 
// (big chunks)
//const int CHUNK_SIZE	= 1024;

#ifdef FREEZE_64
#define CACHE_LINES_IN_CHUNK 	15	// number of cache lines used for arrays of values in a chunk
// little less than a virtual page (for a 64 bit word)
#define VALS_PER_EXIST 			63	// number of bits in one entry of the "exist" array in a chunk
// (for a 64 bit word). One bit used for synchronization.
const int CHUNK_SIZE				// chunk's capacity of values
= (64*(CACHE_LINES_IN_CHUNK-1));// in each cache line enter 64 values (for a 64 bit word)
#else
#define CACHE_LINES_IN_CHUNK 	30	// number of cache lines used for arrays of values in a chunk
// little less than a virtual page (for a 32 bit word)
#define VALS_PER_EXIST 			32	// number of bits in one entry of the "exist" array in a chunk

const int CHUNK_SIZE				// chunk's capacity of values
= (32*(CACHE_LINES_IN_CHUNK-1));// in each cache line enter 32 values
#endif	// FREEZE_64


/*********************************************************************************************************/


/************************************* In-line atomic functionality *****************************************/
inline int atomicCAS(int *adr, int old, int val){
	return  __sync_val_compare_and_swap(adr, old, val);
	// atomic_compare_exchange_strong(adr, old, val);
}

inline u64 atomicCAS(u64 *adr, u64 old, u64 val){
	return __sync_val_compare_and_swap(adr, old, val);
}

inline int atomicInc(int *adr, int inc){
	int res = __sync_fetch_and_add(adr, inc);
	return res;
}

inline int atomicDec(int *adr, int inc){
	int res = __sync_fetch_and_sub(adr, inc);
	return res;
}

inline void *atomicCAS(void **adr, void *old, void *val){
	return __sync_val_compare_and_swap(adr, old, val);
}

inline int atomicOr(int *adr, int mask){
	return __sync_or_and_fetch(adr, mask);
}

inline void membarrier(){
	return __sync_synchronize(); 	// Can be compiled only with GCC
}		
/*********************************************************************************************************/	

#endif  // GLOBALS_H

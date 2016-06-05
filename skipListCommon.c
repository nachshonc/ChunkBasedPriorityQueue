/*
 * skipListCommon.c
 *
 *      Author: elad
 */

#include "skipListCommon.h"
#include "rand.h"

#include <stdlib.h>
#include <assert.h>

/********** AUX FUNCTIONS *************/

int randomLevel() {	
	unsigned val = simRandom();
	int ctr=0; 

	//not accurate because the probability of level==MAX_LEVEL is equal to level==MAX_LEVEL-1. Still reasonable.
	while( (val&1) && ctr<MAX_LEVEL){ //__builtin_ffs work slower.
		ctr++;
		val=val/2;
	}

	assert(ctr>=0 && ctr<=MAX_LEVEL);
	return ctr;
}

ListNode* getListNodeArray(int size) {
	ListNode* res = (ListNode*) malloc(sizeof(ListNode) * (size));
	assert(res != NULL);
	return res;

}

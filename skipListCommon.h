/*
 * skipListCommon.h
 *
 *	constants and structures shared by both skiplists.
 *
 *      Author: Elad Gidron
 */

#ifndef SKIPLISTCOMMON_H_
#define SKIPLISTCOMMON_H_

#ifndef	FALSE
#define	FALSE	(0)
#endif

#ifndef	TRUE
#define	TRUE	(!FALSE)
#endif

#include "listNode.h"

//Max skiplist level
#define MAX_LEVEL 10

#define BOTTOM_LEVEL 0

typedef struct skipList_t {
	ListNode head;
	ListNode tail;
} *SkipList;

/********** AUX FUNCTIONS *************/
int randomLevel();
ListNode* getListNodeArray(int size);


#endif /* SKIPLISTCOMMON_H_ */

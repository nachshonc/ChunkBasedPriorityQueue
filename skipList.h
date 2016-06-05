/*
 * skipList.h
 *
 *      Author: Elad Gidron
 */

#ifndef SKIPLIST_H_
#define SKIPLIST_H_

#include "listNode.h"
#include "skipListCommon.h"

/* creates a new skipList*/
SkipList skipListInit();

/*
 * adds the key and value pair to the skiplist
 * returns 1 on success or 0 if the key was already in the skiplist.
 */
int skipListAdd(SkipList skipList, int key, intptr_t value);

/*
 * removes a key from the skiplist
 * returns 1 on success or 0 if the key wasn't in the skiplist.
 */
int skipListRemove(SkipList skipList, int key);

/*
 * finds a key and relevant value in the skiplist.
 * returns 1 if the key was in the skiplist or 0 if it wasn't.
 * in any case pValue is holding this or previous key value or NULL
 */
int skipListContains(SkipList skipList, int key, intptr_t* pValue);

/*Destroy the skiplist*/
void skipListDestroy(SkipList sl);

#endif /* SKIPLIST_H_ */


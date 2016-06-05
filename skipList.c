/*
 * skipList.c
 *
 * 		A lock-free skiplist,
 * 		Based on the implementation from
 * 		The Art of Multiprocessor Programming
 * 		By Maurice Herlihy & Nir Shavit
 *
 *		With corrections from the book's errata.
 *
 *		Java implementation and errata are available at:
 *		http://www.elsevierdirect.com/companion.jsp?ISBN=9780123705914
 *
 *      Author: Elad Gidron
 */

#include "skipList.h"

#include "atomicMarkedReference.h"

#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <time.h>


int
skipListFind(SkipList skipList, int key, ListNode preds[], ListNode succs[]);

int 
skipListdelete(SkipList skipList, int key, ListNode preds[],ListNode succs[]);



/* creates a new skipList*/
SkipList skipListInit() {
	int i;

	srand(time(NULL));

	SkipList newSkipList = (SkipList) malloc(sizeof(struct skipList_t));
	assert(newSkipList != NULL);
	newSkipList->head = makeSentinelNode(INT_MIN);
	newSkipList->tail = makeSentinelNode(INT_MAX);

	for (i = 0; i <= MAX_LEVEL; i++) {
		newSkipList->head->next[i]
					= NEW_MARKED_REFERENCE(newSkipList->tail, FALSE_MARK);

	}
	return newSkipList;
}




/*
 * adds key to the skiplist
 * returns 1 on success or 0 if the key was already in the skiplist.
 */
int skipListAdd(SkipList skipList, int key, intptr_t value) {

	int topLevel = randomLevel();
	int level;
	ListNode* preds = getListNodeArray(MAX_LEVEL + 1);
	ListNode* succs = getListNodeArray(MAX_LEVEL + 1);

	while (TRUE) {
		//retry:
		int found = skipListFind(skipList, key, preds, succs);

		if (found) {
			free(preds);
			free(succs);
			return FALSE;
		} else {
			ListNode newNode = makeNormalNode(key, topLevel, value);
			for (level = BOTTOM_LEVEL; level <= topLevel; level++) {
				ListNode succ = succs[level];
				SET_ATOMIC_REF(&(newNode->next[level]), succ, FALSE_MARK);
			}
			ListNode pred = preds[BOTTOM_LEVEL];
			ListNode succ = succs[BOTTOM_LEVEL];
			if (  !(REF_CAS(&(pred->next[BOTTOM_LEVEL]), succ, newNode, FALSE_MARK, FALSE_MARK))  ) {
				free(newNode);
				continue;
			}
			for (level = BOTTOM_LEVEL + 1; level <= topLevel; level++) {
				while (TRUE) {
					pred = preds[level];
					succ = succs[level];
					//BUGFIX (Elad): For the case that succ changes when calling find in this loop
					if (succ != GET_REF(newNode->next[level])) {
						int currMark;
						ListNode newNodeSucc = (ListNode)get_mark_and_ref(newNode->next[level], &currMark);
						if (currMark)
							break;
						if (!REF_CAS(&(newNode->next[level]), newNodeSucc, succ, FALSE_MARK, FALSE_MARK)) {
							continue;
						}
					}

					if (REF_CAS(&(pred->next[level]),succ,newNode,FALSE_MARK,FALSE_MARK)) {
						break;
					}
					skipListFind(skipList, key, preds, succs);
				}
			}

			free(preds);
			free(succs);
			return TRUE;
		}
	}
}





/*
 * removes a key from the skiplist
 * returns 1 on success or 0 if the key wasn't in the skiplist.
 */
int skipListRemove(SkipList skipList, int key) {
	int level;
	ListNode* preds = getListNodeArray(MAX_LEVEL + 1);
	ListNode* succs = getListNodeArray(MAX_LEVEL + 1);
	ListNode succ;
	while (TRUE) {
		int found = skipListFind(skipList, key, preds, succs);
		if (!found) {
			free(preds);
			free(succs);
			return FALSE;
		} else {
			ListNode nodeToRemove = succs[BOTTOM_LEVEL];

			for (level = nodeToRemove->topLevel; level >= BOTTOM_LEVEL + 1; level--) {
				int marked = FALSE;
				succ = (ListNode)get_mark_and_ref(nodeToRemove->next[level], &marked);
				while (!marked) {
					REF_CAS(&(nodeToRemove->next[level]), succ, succ, FALSE_MARK, TRUE_MARK);
					succ = (ListNode)get_mark_and_ref(nodeToRemove->next[level], &marked);
				}
			}
			int marked = FALSE;
			succ = (ListNode)get_mark_and_ref(nodeToRemove->next[BOTTOM_LEVEL], &marked);
			while (TRUE) {
				int
				iMarkedIt =
						REF_CAS(&(nodeToRemove->next[BOTTOM_LEVEL]), succ, succ, FALSE_MARK, TRUE_MARK);
				succ = (ListNode)get_mark_and_ref(succs[BOTTOM_LEVEL]->next[BOTTOM_LEVEL], &marked);
				if (iMarkedIt) {
					skipListFind(skipList, nodeToRemove->key, preds, succs);
					free(preds);
					free(succs);
					return TRUE;
				} else if (marked) {
					free(preds);
					free(succs);
					return FALSE;
				}
			}
		}
	}
}





/*
 * finds a key in the skiplist.
 * returns 1 if the key was in the skiplist or 0 if it wasn't.
 * in any case pValue is holding this or previous key value or NULL
 */
int skipListContains(SkipList skipList, int key, intptr_t* pValue) {
	int marked = 0, level;
	*pValue = 0;		// initialize for the case the key is the non-existing minimal 

	while (TRUE) {
		ListNode pred = skipList->head, curr = NULL, succ = NULL;

		for (level = MAX_LEVEL; level >= BOTTOM_LEVEL; level--) {
			curr = (ListNode)GET_REF(pred->next[level]);
			while (TRUE) {
				succ = (ListNode)get_mark_and_ref(curr->next[level], &marked);
				while (marked) {
					curr = (ListNode)GET_REF(curr->next[level]);
					succ = (ListNode)get_mark_and_ref(curr->next[level], &marked);
				}
				if (curr->key < key) {
					*pValue = curr->value;
					pred = curr;
					curr = succ;
				} else {
					break;
				}
			}

		} // end of the for loop going over the levels

		return (curr->key == key);
	} // end of the while loop

}

/*
 * finds an element in the skiplist
 * returns all its predecessors and successors in the preds & succs arrarys.
 * Also makes sure the the first MAX_LEVEL*2 hazard pointers point to the elemesnts in preds & succs
 */
int skipListFind(SkipList skipList, int key, ListNode preds[], ListNode succs[]) {
	int marked, snip, level, retry, currkey;
	ListNode pred = NULL, curr = NULL, succ = NULL;
	while (TRUE) {
		//retry:
		retry = FALSE;
		pred = skipList->head;
		for (level = MAX_LEVEL; level >= BOTTOM_LEVEL; level--) {
			curr = (ListNode)GET_REF(pred->next[level]);

			while (TRUE) {
				succ = (ListNode)get_mark_and_ref(curr->next[level], &marked);
				while (marked) {

					snip
					= REF_CAS(&(pred->next[level]),curr,succ,FALSE_MARK,FALSE_MARK);

					if (!snip) {
						//goto retry
						retry = TRUE;
						break;
					}
					curr = (ListNode)GET_REF(pred->next[level]);
					succ = (ListNode)get_mark_and_ref(curr->next[level], &marked);
				}
				if (retry)
					break;
				//correction from Hazard pointers paper (fig. 9. line 21)
				currkey = curr->key;
				if (curr != GET_REF(pred->next[level])
						|| GET_MARK(pred->next[level])) {
					//goto retry
					retry = TRUE;
					break;
				}
				if (currkey < key) {
					pred = curr;
					curr = succ;
				} else {
					break;
				}
			}
			if (retry)
				break;
			preds[level] = pred;
			succs[level] = curr;
		}
		if (retry)
			continue;
		return (currkey == key);
	}
}

/*Destroy the skiplist*/
void skipListDestroy(SkipList sl) {
	ListNode curr = sl->head;
	ListNode next;

	while (curr != NULL) {
		next = (ListNode)GET_REF(curr->next[BOTTOM_LEVEL]);
		freeListNode(curr);
		curr = next;
	}
	free(sl);
}


//
//  LinkedList.cpp
//  ChunkedSkipList
//
//  Created by Nachshon Cohen O on 6/18/14.
//  Copyright (c) 2014 Nachshon Cohen O. All rights reserved.
//

#include "LinkedList.h"

#define CAS(adr, exp, val) __sync_bool_compare_and_swap(adr, exp, val)

bool find(LinkedList *entryHead, Entry ***outprev, Entry **outcur, int key);

static inline bool isDeleted(Entry* p) {
	return ( (long)p & 1l);
}

static inline Entry* clearDeleted(Entry *p){
	return (Entry*)(((long)p)&~1l);
}


/*****************************************************************************/
bool ListSearch(LinkedList* entryHead, int key) {
	Entry *cur, **prev;
	if (find(entryHead, &prev, &cur,key)) {
		return true;
	}
	return false;
}
/*****************************************************************************/
/*****************************************************************************/
bool ListInsert(LinkedList* entryHead,int key) {
	Entry *cur, **prev;
	while (true) {

		if (find(entryHead, &prev, &cur, key)) 	// key exists
			return false; 

		Entry* newEntry = new Entry();			// create entry
		newEntry->key	= key;
		newEntry->nextEntry = cur;

		if (CAS(prev, cur, newEntry)) 			//connect
			return true;
	} // end of while
}
/*****************************************************************************/
/*****************************************************************************/
int ListDelMin(LinkedList *head){
	Entry *h;
	Entry *next;

	do{
		h = head->head;
		if (h==NULL) return -1;
		next = h->nextEntry;
	} while(!CAS(&head->head, h, next));

	return h->key;
}
bool find(LinkedList *entryHead, Entry ***outprev, Entry **outcur, int key) {
	int ckey;
	Entry **prev, *cur, *next;

	try_again:
	prev = &entryHead->head;
	cur = *(prev);

	while (cur != NULL) {
		next = cur->nextEntry;

		if(isDeleted(next)) {
			if (!CAS(prev, cur, clearDeleted(next)))
				goto try_again; 		// disconnect of the deleted entry failed, try again

			cur = clearDeleted(next);
		} else {
			ckey = cur->key;

			if (*(prev) != cur) {
				goto try_again;
			}

			if (ckey >= key) {
				*outprev = prev;
				*outcur	 = cur;
				return (ckey == key); 	//compare search key
			}

			prev = &(cur->nextEntry);
			cur = next;
		}
	} //end of while
	*outprev=prev;
	*outcur=cur;
	return false;
}

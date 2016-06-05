//
//  LinkedList.h
//  ChunkedSkipList
//
//  Created by Nachshon Cohen O on 6/18/14.
//  Copyright (c) 2014 Nachshon Cohen O. All rights reserved.
//

#ifndef __ChunkedSkipList__LinkedList__
#define __ChunkedSkipList__LinkedList__

#include <iostream>


typedef struct Entry_t {
	int key;
	struct Entry_t* nextEntry;
} Entry;


typedef struct {
	Entry *head;
} LinkedList;


int ListDelMin(LinkedList *head);
bool ListInsert(LinkedList* entryHead, int key);


#endif /* defined(__ChunkedSkipList__LinkedList__) */

/*
 * listNode.h
 *
 *      Author: Elad Gidron
 */

#ifndef LISTNODE_H_
#define LISTNODE_H_

#include <stdint.h>

#include "atomicMarkedReference.h"

typedef struct listNode_t {
	int key;
	intptr_t value;
	int topLevel;
	markable_ref next[];
} *ListNode;

ListNode makeSentinelNode(int key);
ListNode makeNormalNode(int key, int height, intptr_t value);
void freeListNode(ListNode node);

#endif /* LISTNODE_H_ */

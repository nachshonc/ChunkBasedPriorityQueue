/*
 * listNode.c
 *
 *      Author: Elad Gidron
 */


#include "listNode.h"
#include "skipListCommon.h"
#include <stdlib.h>
#include <assert.h>

ListNode makeSentinelNode(int key) {
	return makeNormalNode(key, MAX_LEVEL, 0);
}

ListNode makeNormalNode(int key, int height, intptr_t value) {
	int i;
	ListNode newNode = (ListNode)malloc(sizeof (struct listNode_t) + (sizeof(markable_ref) * (height+1)));
	assert(newNode != NULL);
	newNode->key = key;
	newNode->value = value;

	for (i = 0; i <= height; i++)
		newNode->next[i] = (markable_ref)NULL;
	newNode->topLevel = height;

	return newNode;
}

void freeListNode(ListNode node) {
	free(node);
}

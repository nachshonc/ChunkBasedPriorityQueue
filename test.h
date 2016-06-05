//
//  test.h
//  ChunkedSkipList
//
//  Created by Nachshon Cohen O on 6/18/14.
//  Copyright (c) 2014 Nachshon Cohen O. All rights reserved.
//

#ifndef __ChunkedSkipList__test__
#define __ChunkedSkipList__test__

#include <iostream>

#include "ChunkedPriorityQueue.h"

typedef struct privateThreadInfo{

	ChunkedPriorityQueue* pq;
	int* 			values;			// used for debugging only
	int num;	
	int 			id;				// the thread ID
	unsigned int    num_ops;
	int				insertCnt;
	int				insInFirstCnt;
	int				deleteCnt;
	int             eleminCnt;

	bool			throughput;
	int 			key;
} ThrInf;

#endif /* defined(__ChunkedSkipList__test__) */

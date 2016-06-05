//
//  debug.h
//  ChunkedSkipList
//
//  Created by Nachshon Cohen O on 6/18/14.
//  Copyright (c) 2014 Nachshon Cohen O. All rights reserved.
//

#ifndef ChunkedSkipList_debug_h
#define ChunkedSkipList_debug_h

#include <assert.h>
#include <stdio.h>

#define SKIP_LIST		// enable this macro to measure with the skip-list
//#define FLAG_RELEASE		// enable smart schedulin
#define ELIMINATION     // enable for elimination from buffer
//#define FIRST_CHUNK_EXPLICIT_FREEZE

//#define DEBUG
//#define DEBUG2

#ifdef DEBUG

#define _assert assert
#define PRINT(...) printf(__VA_ARGS__)			// use printf and not cout because of the multi-threading
#define DEB(...) __VA_ARGS__

#else

#define _assert(B)
#define PRINT(...)
#define DEB(...)

#endif



#ifdef DEBUG2

#define PRINT2(...) printf(__VA_ARGS__)
#define DEB2(...) __VA_ARGS__

#else

#define PRINT2(...)
#define DEB2(...)

#endif

#define TODO(...) 


#endif

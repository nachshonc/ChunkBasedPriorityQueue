/*
 * atomicMarkedReference.h
 *
 * same behavior like Java's AtomicMarkableReference:
 *
 * http://download.oracle.com/javase/1.5.0/docs/api/java/util/concurrent/atomic/AtomicMarkableReference.html
 *      Author: Elad Gidron
 */

#ifndef ATOMICMARKEDREFERENCE_H_
#define ATOMICMARKEDREFERENCE_H_

#include <stdio.h>
#include <stdlib.h>

#include "atomic.h"

typedef size_t markable_ref;

#define REF_MASK (~(0x1))
#define TRUE_MARK 0x1
#define FALSE_MARK 0x0
#define LSB 0x1

#define NEW_MARKED_REFERENCE(addr, mark) ((markable_ref)addr | mark)

#define GET_MARK(m_ref) ((int)(m_ref & LSB))
#define GET_REF(m_ref) ((void*)(m_ref & REF_MASK))

static inline void* get_mark_and_ref(markable_ref m_ref, int* mark)
{
	*mark = GET_MARK(m_ref);
	return GET_REF(m_ref);
}

//adds mark to a pointer
#define ADD_MARK(addr, mark) ((markable_ref)(GET_REF((markable_ref)addr)) | mark)

//atomic actions:
#define SET_ATOMIC_REF(ptr, newAddr, newMark) \
		(*ptr = ADD_MARK(newAddr, newMark))

#define REF_CAS(ptr,oldaddr,newaddr,oldmark,newmark) (CAS(ptr, ADD_MARK(oldaddr, oldmark), ADD_MARK(newaddr, newmark)))

#endif

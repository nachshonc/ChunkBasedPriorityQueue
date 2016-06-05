/*
 * atomic.h
 *
 * wrappers for the glib atomic functions.
 *
 *      Author: Elad Gidron
 */

#ifndef MYATOMIC_H_
#define MYATOMIC_H_


//For X86 Systems only:
#define X_86_ARCH

#ifdef X_86_ARCH
//#include <glib.h>
#define CAS(ptr, oldval, newval) \
		__sync_val_compare_and_swap(ptr, oldval, newval)
/*g_atomic_pointer_compare_and_exchange \
(((gpointer*)(ptr)),((gpointer)(oldval)),((gpointer)(newval)))*/


//For solaris systems:
#else
#include <atomic.h>

#define CAS(ptr, oldval, newval) \
		(atomic_cas_ptr\
				(((void*)ptr),((void*)oldval),((void*)newval)) == (void*)oldval)

#endif

#endif /* CAS_H_ */

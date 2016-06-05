#ifndef _RAND_H_

#define _RAND_H_
#define SIM_RAND_MAX         32767

/*
 * This random generators are implementing 
 * by following POSIX.1-2001 directives.
 */

//__thread unsigned long nextr = 1;
extern __thread unsigned long nextr;

inline static long simRandom(void) {
    nextr = nextr * 1103515245 + 12345;
    return((unsigned)(nextr/65536));
}

inline static void simSRandom(unsigned long seed) {
    nextr = seed;
}

/*
 * In Numerical Recipes in C: The Art of Scientific Computing 
 * (William H. Press, Brian P. Flannery, Saul A. Teukolsky, William T. Vetterling;
 *  New York: Cambridge University Press, 1992 (2nd ed., p. 277))
 */
inline static long simRandomRange(long low, long high) {
    return low + (long) ( ((double) high)* (simRandom() / (SIM_RAND_MAX + 1.0)));
}

#endif

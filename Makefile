
CC 			= g++
CC_FLAGS 	= -O3 -pthread -Wall
LIBS 		= -lrt
LDFLAGS	        = -lpthread `pkg-config --libs gsl`

OBJECTS = Atomicable.o LinkedList.o ChunkedPriorityQueue.o listNode.o skipListCommon.o skipList.o
TEST_OBJS = test.o

####################### ALL ######################
all: ChunkedPQ
####################### EXECUTABLE ######################

ChunkedPQ: ${OBJECTS} ${TEST_OBJS}
	${CC} ${CC_FLAGS} ${OBJECTS} ${TEST_OBJS} -o $@ ${LDFLAGS} ${LIBS}

###################### OBJECTS ######################
Atomicable.o: Atomicable.cpp Atomicable.h
	${CC} ${CC_FLAGS} -c $*.cpp -o $@ ${LIBS}

listNode.o: listNode.c listNode.h skipListCommon.h
	${CC} ${CC_FLAGS} -c $*.c -o $@ ${LIBS}
	
skipListCommon.o: skipListCommon.c listNode.h skipListCommon.h rand.h
	${CC} ${CC_FLAGS} -c $*.c -o $@ ${LIBS}
	
skipList.o: skipList.c skipList.h listNode.h skipListCommon.h atomicMarkedReference.h
	${CC} ${CC_FLAGS} -c $*.c -o $@ ${LIBS}
	
ChunkedPriorityQueue.o: ChunkedPriorityQueue.cpp globals.h ChunkedPriorityQueue.h test.h Atomicable.h skipList.h
	${CC} ${CC_FLAGS} -c $*.cpp -o $@ ${LIBS}
	
LinkedList.o: LinkedList.cpp LinkedList.h
	${CC} ${CC_FLAGS} -c $*.cpp -o $@ ${LIBS}

	
###################### TEST_OBJS ######################		
test.o: test.cpp ChunkedPriorityQueue.h test.h debug.h LinkedList.h rand.h globals.h Atomicable.h
	${CC} ${CC_FLAGS} -c $*.cpp -o $@ ${LDFLAGS} ${LIBS}

######################Clean######################

clean:
	rm -f ChunkedPQ *.o
	
	

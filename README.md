Code for the paper:
"CBPQ: High Performance Lock-Free Priority Queue", Anastasia Braginsky, Nachshon Cohen and Erez Petrank, EuroPar, 2016. 

Provides a priority queue implementation that is highly performant and scalable. 
Mostly uses fetch_and_add instead of the traditional CAS to improve performance and scalability. 
Also supports eliminations. 

ChunkedPriorityQueue.h: the priority queue class file. 
test.cpp: code for testing the priority queue's performance. 


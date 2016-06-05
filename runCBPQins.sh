echo -n $1; echo -n " ";

for ((i=1;i<=$2;i++)); do

	timeout 120s ./ChunkedPQ $1 1000000 1 1 | grep INSER | cut -d" " -f15 | cut -d"," -f1; 


done | column


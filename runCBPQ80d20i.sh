echo -n $1; echo -n " ";

for ((i=1;i<=$2;i++)); do

	timeout 120s ./ChunkedPQ $1 10000000 3 1 | grep MIX | cut -d" " -f13 | cut -d"," -f1; 


done | column


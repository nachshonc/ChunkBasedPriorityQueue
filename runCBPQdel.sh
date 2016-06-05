echo -n $1; echo -n " ";

for ((i=1;i<=$2;i++)); do

	timeout 120s ./ChunkedPQ $1 10000000 2 1 | grep DEL | cut -d" " -f12 | cut -d"," -f1; 


done | column


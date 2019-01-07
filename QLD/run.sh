#! /bin/bash

echo "Running $1 experiments..."

for i in `seq 1 $1`;
do
  python bootstrap.py --flag=bootstrap/run$i &
#   echo $i
done
wait

echo "Finished."
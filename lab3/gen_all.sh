#!/bin/bash
mkdir -p data
for n in 200 400 800 1200 1600 2000; do
  python3 generate_data.py $n data/input_${n}.txt
done

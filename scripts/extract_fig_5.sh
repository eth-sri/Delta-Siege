#!/bin/bash
for N in 500 1_250 2_500 5_000 12_500 25_000
do
	echo $N
	python scripts/run_benchmarks.py --mechanism mst_internal --epsilon 3.0 --delta 1e-6 --n $N $@
done

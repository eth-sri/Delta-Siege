#!/bin/bash
for mech in laplace_opendp laplace_ibm laplace_pydp laplace_inversion gauss_opendp gauss_pydp gauss_opacus gauss_ibm gauss_ibm_analytic gauss_ibm_discrete gauss_ziggurat gauss_polar gauss_boxmuller aim_internal mst_internal
do
	echo $mech
	python scripts/run_benchmarks.py --mechanism $mech $@
done

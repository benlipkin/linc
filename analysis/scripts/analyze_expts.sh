#!/bin/bash

set -e
source activate linc

mkdir -p ../tables
mkdir -p ../figures

python -m compile_results
python -m plot_figures

touch ../analyze.done

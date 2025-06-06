#!/bin/bash
NETWORK=$1
N=1000
KAVE=6.0
K=$2
TMAX=50
TINTERVAL=0.3
NUM_RUNS=100

if [ "$7" = "pos" ]; then
    EPS_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.005 0.015 0.025 0.035 0.045 0.055 0.065 0.075 0.085 0.095)
else
    EPS_VALUES=(0.0 -0.01 -0.02 -0.03 -0.04 -0.05 -0.06 -0.07 -0.08 -0.09 -0.1 -0.005 -0.015 -0.025 -0.035 -0.045 -0.055 -0.065 -0.075 -0.085 -0.095)
fi

TOTAL_SIMS=${#EPS_VALUES[@]}
CURRENT_SIM=0

for EPS in "${EPS_VALUES[@]}"; do
    CURRENT_SIM=$((CURRENT_SIM + 1))
    
    echo -n "[$CURRENT_SIM/$TOTAL_SIMS] Running eps=$EPS ... "

    python run_kuramoto.py \
        --network $NETWORK \
        --N $N \
        --kave $KAVE \
        --K $K \
        --eps $EPS \
        --tmax $TMAX \
        --t_interval $TINTERVAL \
        --num_runs $NUM_RUNS
    
    python run_kuramoto.py \
        --network $NETWORK \
        --N $N \
        --kave $KAVE \
        --K $K \
        --eps $EPS \
        --tmax $TMAX \
        --t_interval $TINTERVAL \
        --num_runs $NUM_RUNS \
        --random

done


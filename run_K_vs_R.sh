#!/bin/bash
NETWORK=$1
N=10
KAVE=6.0
TMAX=5
TINTERVAL=0.3
NUM_RUNS=10

# K_VALUES=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
K_VALUES=(0.2 0.4 0.6 0.8 1.0)
EPS_VALUES=(0.0 0.03 -0.03 0.05 -0.05 0.07 -0.07)

TOTAL_SIMS=$((${#K_VALUES[@]} * ${#EPS_VALUES[@]}))
CURRENT_SIM=0

for K in "${K_VALUES[@]}"; do
    for EPS in "${EPS_VALUES[@]}"; do
        CURRENT_SIM=$((CURRENT_SIM + 1))
        
        echo -n "[$CURRENT_SIM/$TOTAL_SIMS] Running K=$K, eps=$EPS ... "
        
        python run_kuramoto.py \
               --network $NETWORK \
               --N $N \
               --kave $KAVE \
               --K $K \
               --eps $EPS \
               --tmax $TMAX \
               --t_interval $TINTERVAL \
               --num_runs $NUM_RUNS
    done
done


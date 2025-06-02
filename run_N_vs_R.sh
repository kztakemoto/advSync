#!/bin/bash
NETWORK=$1
KAVE=6.0
K=$2
TMAX=50
TINTERVAL=0.3
NUM_RUNS=100

N_VALUES=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2200 2500 2800 3000 3200 3500 3800 4000)

if [ "$6" = "pos" ]; then
    EPS_VALUES=(0.0 0.03 0.05 0.07)
else
    EPS_VALUES=(0.0 -0.03 -0.05 -0.07)
fi

TOTAL_SIMS=$((${#N_VALUES[@]} * ${#EPS_VALUES[@]}))
CURRENT_SIM=0

for N in "${N_VALUES[@]}"; do
    for EPS in "${EPS_VALUES[@]}"; do
        CURRENT_SIM=$((CURRENT_SIM + 1))
        
        echo -n "[$CURRENT_SIM/$TOTAL_SIMS] Running N=$N eps=$EPS... "
        
        python run_kuramoto.py \
                --network $NETWORK \
                --N $N \
                --kave $KAVE \
                --K $K \
                --eps $EPS \
                --tmax $TMAX \
                --t_interval $TINTERVAL \
                --num_runs $NUM_RUNS \
        
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
done


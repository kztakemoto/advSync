# advSync

This repository contains data and code for the research paper "Adversarial control of synchronization in complex oscillator networks."

## Terms of Use

This project is licensed under the MIT License. When using this code, please cite our paper:


Nagahama Y, Miyazato K & Takemoto K (2025) **Adversarial Control of Synchronization in Complex Oscillator Networks.**

## Requirements

- Python 3.11

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Kuramoto Model Simulations

#### Basic Usage

To run simulations on Erdős-Rényi networks:

```bash
python run_kuramoto.py --network ER
```

**Default parameters:** $N=1000$, $\langle k \rangle = 6$, $K=0.4$, $\epsilon=0.05$ (see `run_kuramoto.py` for details)

#### Network Types

**Model networks:**
- `--network BA`: Barabási-Albert networks
- `--network WS`: Watts-Strogatz networks

**Real-world networks:**
- `--network power-1138-bus`: Power network
- `--network bn-mouse`: Brain network

### K-R Curve Analysis

#### Generate K-R Data

To run simulations for obtaining K-R curves on Erdős-Rényi networks:

```bash
bash run_K_vs_R.sh ER
```

For other networks, replace `ER` with the appropriate network argument from the list above.

#### Plot K-R Curves

```bash
python plot.py --plot_type K_vs_R --network ER
```

### $\epsilon$-$R$ Curve Analysis

#### Generate $\epsilon$-R Data

To run simulations for obtaining $\epsilon$-R curves on Erdős-Rényi networks with $K=0.3$:

```bash
bash run_eps_vs_R.sh ER 0.3
```

The network arguments are the same as described above.

#### Plot $\epsilon$-R Curves

```bash
python plot_R.py --plot_type eps_vs_R --network ER
```

### N-R Curve Analysis

#### Generate N-R Data

To run simulations for obtaining N-R curves on Erdős-Rényi networks with $K=0.3$:

```bash
bash run_N_vs_R.sh ER 0.3
```

The network arguments are the same as described above.

#### Plot N-R Curves

```bash
python plot_R.py --plot_type N_vs_R --network ER
```

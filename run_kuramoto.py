import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit
import argparse
from multiprocessing import Pool, cpu_count
import time
import os
import pandas as pd

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER), Barabasi-Albert (BA), and Watts-Strogatz (WS) networks')
parser.add_argument('--N', type=int, default=1000, help='number of nodes')
parser.add_argument('--kave', type=float, default=6.0, help='average degree in model networks')
parser.add_argument('--K', type=float, default=0.4, help='coupling constant')
parser.add_argument('--eps', type=float, default=0.05, help='perturbation strength for attacks')
parser.add_argument('--tmax', type=float, default=50.0, help='maximum simulation time')
parser.add_argument('--attack_type', type=str, default='node', help='attack type')
parser.add_argument('--t_interval', type=float, default=0.3, help='perturbation interval')
parser.add_argument('--random', action='store_true', help='perform random attacks with strength eps')
parser.add_argument('--seed', type=int, default=123, help='random seed for reproducibility')
parser.add_argument('--num_runs', type=int, default=100, help='number of parallel simulation runs (iterations)')
parser.add_argument('--output_dir', type=str, default='results', help='directory to save results')
parser.add_argument('--attack_start', type=float, default=0.0, help='time to start the attack')
parser.add_argument('--attack_end', type=float, default=None, help='time to end the attack (if None, attacks continue until tmax)')
args = parser.parse_args()

real_network_data = ['power-1138-bus', 'bn-mouse']

# Kuramoto model
@jit(nopython=True)
def kuramoto_numba(theta, K, omega, adj_matrix):
    N = len(theta)
    dtheta = np.zeros(N)
    
    for i in range(N):
        sum_sin = 0.0
        for j in range(N):
            if adj_matrix[i, j] != 0:
                sum_sin += adj_matrix[i, j] * np.sin(theta[j] - theta[i])
        
        dtheta[i] = omega[i] + K * sum_sin
    
    return dtheta


# wrapper for solve_ivp
def kuramoto_wrapper(t, theta, K, omega, adj_matrix):
    return kuramoto_numba(theta, K, omega, adj_matrix)

# Order parameter calculation
def calculate_order_parameter(theta):
    """Calculate the Kuramoto order parameter R"""
    return np.abs(np.mean(np.exp(1j * theta)))

# Generate network based on type and parameters
def generate_network(network_type, N, kave, seed):
    """Generate a network based on specified type and parameters"""
    if network_type == 'BA':
        # Barabasi-Albert model
        g = nx.barabasi_albert_graph(N, int(kave / 2), seed=seed)
    elif network_type == 'ER':
        # Erdos-Renyi model
        g = nx.gnm_random_graph(N, int(kave * N / 2), directed=False, seed=seed)
    elif network_type == 'WS':
        # Watts-Strogatz model
        pws = 0.05
        g = nx.watts_strogatz_graph(N, int(kave), pws, seed=seed)
    elif network_type in real_network_data:
        df = pd.read_csv(f"./network_data/{network_type}.txt", sep='\s+', header=None)
        g = nx.from_pandas_edgelist(df, source=0, target=1)
        g = nx.Graph(g)
        g.remove_edges_from(nx.selfloop_edges(g))
        lcc = max(nx.connected_components(g), key=len)
        g = g.subgraph(lcc)
    else:
        raise ValueError(f"Invalid network type: {network_type}")
    
    return nx.to_numpy_array(g)

def run_kuramoto_simulation(params):
    """
    Run a single Kuramoto simulation and return order parameter time series
    
    Parameters:
    -----------
    params : tuple
        (run_id, dt, network_type, N, kave, eps, K, tmax, t_interval, attack_type, random_attack)
    
    Returns:
    --------
    times : ndarray
        Time points
    R_values : ndarray
        Order parameter values at each time point
    run_id : int
        ID of this simulation run
    """
    (run_id, dt, network_type, N, kave, eps, K, tmax, t_interval, attack_type, 
     random_attack, attack_start, attack_end, seed) = params
    
    # Set unique seed for this run
    run_seed = seed + run_id * 100
    np.random.seed(run_seed)
    
    # Generate new network for each run
    adj_matrix = generate_network(network_type, N, kave, seed=run_seed)

    if network_type in real_network_data:
        N = len(adj_matrix)
    
    # Generate new natural frequencies with std=1.0
    omega = np.random.normal(0, 1.0, N)
    
    # Generate new initial phases
    theta0 = np.random.uniform(0, 2*np.pi, N)
    
    # pre-running for JIT
    _ = kuramoto_numba(theta0, 0.1, omega, adj_matrix)
    
    t_current = 0
    theta_current = theta0.copy()
    adj_matrix_current = adj_matrix.copy()
    
    # Initialize data storage
    times = []
    R_values = []
    
    while t_current < tmax:
        t_end = np.round(min(t_current + t_interval, tmax), 6)
        t_eval = np.round(np.arange(t_current, t_end, dt), 6)

        sol = solve_ivp(
            kuramoto_wrapper,
            [t_current, t_end],
            theta_current,
            t_eval=t_eval,
            args=(K, omega, adj_matrix_current)
        )
        
        # Calculate order parameter for each time point
        for i, t in enumerate(sol.t):
            times.append(t)
            R = calculate_order_parameter(sol.y[:, i])
            R_values.append(R)

        theta_current = np.mod(sol.y[:, -1], 2 * np.pi)

        del sol

        # add perturbation
        actual_attack_end = tmax if attack_end is None else attack_end
        if t_end < tmax and eps != 0.0 and attack_start <= t_current < actual_attack_end:
            psi = np.angle(np.mean(np.exp(1j * theta_current)))

            if attack_type == 'node':
                if random_attack:
                    theta_current = theta_current + np.random.choice([-eps, eps], size=len(theta_current))
                else:
                    theta_current = theta_current + eps * np.sign(np.sin(psi - theta_current))
            
            else:
                raise ValueError(f"Invalid attck type: {attack_type}.")

        t_current = t_end
    
    return np.array(times), np.array(R_values), run_id

def run_parallel_simulations(num_runs, dt=0.01):
    """
    Run multiple Kuramoto simulations in parallel with different network structures
    and natural frequencies for each run
    
    Parameters:
    -----------
    num_runs : int
        Number of simulation runs
    dt : float
        Time step for the simulation
    
    Returns:
    --------
    results : list
        List of (times, R_values, run_id) tuples for each run
    """
    # Use all available CPU cores
    num_processes = cpu_count()
    
    # Prepare parameters for each run
    run_params = []
    for run_id in range(num_runs):
        run_params.append((
            run_id, 
            dt,
            args.network,
            args.N,
            args.kave,
            args.eps,
            args.K,
            args.tmax,
            args.t_interval,
            args.attack_type,
            args.random,
            args.attack_start,
            args.attack_end,
            args.seed
        ))
    
    # Run simulations in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_kuramoto_simulation, run_params)
    
    return results, num_processes

if __name__ == "__main__":
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate filename based on parameters
    if args.eps == 0:
        if args.network in real_network_data:
            filename_base = f"results_{args.network}_K{args.K}_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}"
        else:
            filename_base = f"results_{args.network}_N{args.N}_kave{args.kave}_K{args.K}_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}"
    else:
        if args.network in real_network_data:
            filename_base = f"results_{args.network}_K{args.K}_tmax{args.tmax}_{args.attack_type}_attack_eps{args.eps}_interval{args.t_interval}_nbruns{args.num_runs}_seed{args.seed}"
        else:
            filename_base = f"results_{args.network}_N{args.N}_kave{args.kave}_K{args.K}_tmax{args.tmax}_{args.attack_type}_attack_eps{args.eps}_interval{args.t_interval}_nbruns{args.num_runs}_seed{args.seed}"
        
        if args.attack_start > 0:
            filename_base += f"_start{args.attack_start}"
        if args.attack_end is not None:
            filename_base += f"_end{args.attack_end}"

        if args.random:
            filename_base += "_random"

    combined_csv_filename = os.path.join(args.output_dir, f"{filename_base}_all_runs.csv")
    # Check if results already exist
    if os.path.exists(combined_csv_filename):
        print(f"Results already exist for this configuration at: {combined_csv_filename}")
        print("Skipping calculation. Delete the file if you want to recalculate.")
        exit(0)
    
    # Set base random seed
    np.random.seed(args.seed)
    
    # Run parallel simulations
    results, num_processes = run_parallel_simulations(num_runs=args.num_runs)
    
    # Print run configuration
    print(f"Completed {args.num_runs} Kuramoto simulations with:")
    print(f"- Network type: {args.network}")
    print(f"- Nodes: {args.N}")
    print(f"- Average degree: {args.kave}")
    print(f"- Coupling constant K: {args.K}")
    if args.eps != 0:
        print(f"- Perturbation strength eps: {args.eps}")
        print(f"- Attack type: {args.attack_type}")
    print(f"- Different network and frequency realizations used for each run")
    print(f"- Parallel processes: {num_processes} (all available cores)")
    
    # Save a combined CSV with all runs
    # First, use the first run's time points as reference
    ref_times = results[0][0]
    
    # Create a combined Dict starting with the time column
    data_dict = {'time': ref_times}
    
    # Add each run's order parameter as a new column
    for times, R_values, run_id in results:
        # If time points differ, interpolate to match reference times
        if len(times) != len(ref_times) or not np.allclose(times, ref_times):
            from scipy.interpolate import interp1d
            f = interp1d(times, R_values, bounds_error=False, fill_value="extrapolate")
            interpolated_R = f(ref_times)
            data_dict[f'R_run{run_id}'] = interpolated_R
        else:
            data_dict[f'R_run{run_id}'] = R_values
    
    # Create dataframe
    combined_df = pd.DataFrame(data_dict)
    
    # Calculate and add mean R value
    R_columns = [col for col in combined_df.columns if col.startswith('R_run')]
    combined_df['R_mean'] = combined_df[R_columns].mean(axis=1)
    
    # Save combined results
    combined_df.to_csv(combined_csv_filename, index=False)
    print(f"Saved combined results to {combined_csv_filename}")

    # Create a simple plot to check convergence
    plt.plot(ref_times, combined_df['R_mean'], 'b-', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Order Parameter R (Mean)', fontsize=12)
    plt.ylim(0,1)
    title = f'Kuramoto Model Synchronization ({args.network} Network, N={args.N}, kave={args.kave}, K={args.K}, eps={args.eps})'
    
    # Save the figure
    plt_filename = os.path.join(args.output_dir, f"R_vs_t_{filename_base}.png")
    plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
    print(f"Saved R vs time plot to {plt_filename}")
    
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import re

real_network_data = ['power-1138-bus', 'bn-mouse']

def parse_args():
    parser = argparse.ArgumentParser(description='Plot Kuramoto simulation results')
    parser.add_argument('--results_dir', type=str, default='results', help='directory containing result files')
    parser.add_argument('--network', type=str, default='ER', help='network type (ER, BA, WS)')
    parser.add_argument('--N_values', type=str, default=None, help='comma-separated list of N values to include')
    parser.add_argument('--kave_values', type=str, default=None, help='average degree')
    parser.add_argument('--tmax', type=float, default=50.0, help='maximum simulation time')
    parser.add_argument('--attack_type', type=str, default='node', help='attack type (node)')
    parser.add_argument('--t_interval_values', type=str, default=None, help='perturbation interval')
    parser.add_argument('--num_runs', type=int, default=100, help='number of simulation runs')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--eval_time', type=float, default=20.0, help='specific time point to evaluate R (default: end of simulation)')
    parser.add_argument('--output_dir', type=str, default='figures', help='directory to save plots')
    parser.add_argument('--K_values', type=str, default=None, help='comma-separated list of K values to include')
    parser.add_argument('--eps_values', type=str, default=None, help='comma-separated list of eps values to include')
    parser.add_argument('--time_window', type=int, default=5, help='number of time points at the end to average for final R value')
    parser.add_argument('--plot_type', type=str, default='K_vs_R', help='type of plot to generate (K_vs_R, eps_vs_R, or N_vs_R)')
    return parser.parse_args()

def extract_params_from_filename(filename):
    """Extract parameters from the result filename"""
    params = {}
    
    # Extract K value
    k_match = re.search(r'_K([\d\.]+)_', filename)
    if k_match:
        params['K'] = float(k_match.group(1))

    # Extract N value
    n_match = re.search(r'_N([\d\.]+)_', filename)
    if n_match:
        params['N'] = float(n_match.group(1))

    # Extract kave value
    kave_match = re.search(r'_kave([\d\.]+)_', filename)
    if kave_match:
        params['kave'] = float(kave_match.group(1))

    # Extract t_interval value
    t_interval_match = re.search(r'_interval([\d\.]+)_', filename)
    if t_interval_match:
        params['t_interval'] = float(t_interval_match.group(1))
    else:
        params['t_interval'] = -999
    
    # Extract eps value
    eps_match = re.search(r'_eps(-?[\d\.]+)_', filename)
    if eps_match:
        params['eps'] = float(eps_match.group(1))
    else:
        params['eps'] = 0  # No perturbation

    # Extract start time value
    start_match = re.search(r'_start([\d\.]+)_', filename)
    if start_match:
        params['start_time'] = float(start_match.group(1))
    else:
        params['start_time'] = 0
    
    # Extract end time value
    end_match = re.search(r'_end([\d\.]+)_', filename)
    if end_match:
        params['end_time'] = float(end_match.group(1))
    else:
        params['end_time'] = -999.0
    
    # Extract attack type
    if '_node_attack_' in filename:
        params['attack_type'] = 'node'
    else:
        params['attack_type'] = 'none'
    
    # Check if this is a random perturbation
    if '_random' in filename:
        params['is_random'] = True
    else:
        params['is_random'] = False
    
    return params

def find_result_files(args):
    """Find all matching result files based on the specified parameters"""
    # Get a list of all mean CSV files in the results directory
    if args.K_values:
        K_list = [float(x) for x in args.K_values.split(',')]
    else:
        K_list = None
    
    if args.eps_values:
        eps_list = [float(x) for x in args.eps_values.split(',')]
    else:
        eps_list = None

    if args.N_values:
        N_list = [float(x) for x in args.N_values.split(',')]
    else:
        N_list = None
    
    if args.kave_values:
        kave_list = [float(x) for x in args.kave_values.split(',')]
    else:
        kave_list = None

    if args.t_interval_values:
        t_interval_list = [float(x) for x in args.t_interval_values.split(',')] + [-999]
    else:
        t_interval_list = None

    all_files = []
    
    # Pattern for eps=0 case
    if args.network in real_network_data:
        base_pattern = f"results_{args.network}_K*_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}"
    else:
        base_pattern = f"results_{args.network}_N*_kave*_K*_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}"
    pattern1 = os.path.join(args.results_dir, f"{base_pattern}_all_runs.csv")
    
    # Pattern for targeted perturbation
    if args.network in real_network_data:
        attack_pattern = f"results_{args.network}_K*_tmax{args.tmax}_{args.attack_type}_attack_eps*_interval*_nbruns{args.num_runs}_seed{args.seed}"
    else:
        attack_pattern = f"results_{args.network}_N*_kave*_K*_tmax{args.tmax}_{args.attack_type}_attack_eps*_interval*_nbruns{args.num_runs}_seed{args.seed}"
    pattern2 = os.path.join(args.results_dir, f"{attack_pattern}*_all_runs.csv")
    
    # Pattern for random perturbation
    if args.network in real_network_data:
        random_attack_pattern = f"results_{args.network}_K*_tmax{args.tmax}_{args.attack_type}_attack_eps*_interval*_nbruns{args.num_runs}_seed{args.seed}_random"
    else:
        random_attack_pattern = f"results_{args.network}_N*_kave*_K*_tmax{args.tmax}_{args.attack_type}_attack_eps*_interval*_nbruns{args.num_runs}_seed{args.seed}_random"
    pattern3 = os.path.join(args.results_dir, f"{random_attack_pattern}*_all_runs.csv")
    
    # Find all files matching any pattern
    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)
    files3 = glob.glob(pattern3)
    all_files = files1 + files2 + files3
    
    # Filter by K values if specified
    if K_list is not None:
        filtered_files = []
        for file in all_files:
            params = extract_params_from_filename(file)
            if params['K'] in K_list:
                filtered_files.append(file)
        all_files = filtered_files
    
    # Filter by eps values if specified
    if eps_list is not None:
        filtered_files = []
        for file in all_files:
            params = extract_params_from_filename(file)
            if params['eps'] in eps_list:
                filtered_files.append(file)
        all_files = filtered_files
    
    # Filter by N values if specified
    if N_list is not None:
        filtered_files = []
        for file in all_files:
            params = extract_params_from_filename(file)
            if params['N'] in N_list:
                filtered_files.append(file)
        all_files = filtered_files
    
    # Filter by N values if specified
    if kave_list is not None:
        filtered_files = []
        for file in all_files:
            params = extract_params_from_filename(file)
            if params['kave'] in kave_list:
                filtered_files.append(file)
        all_files = filtered_files

    # Filter by t_interval values if specified
    if t_interval_list is not None:
        filtered_files = []
        for file in all_files:
            params = extract_params_from_filename(file)
            if params['t_interval'] in t_interval_list:
                filtered_files.append(file)
        all_files = filtered_files
    
    return all_files

def calculate_R_at_time(csv_file, eval_time=None, time_window=5):
    """
    Calculate the R value at a specific time point or at the end of simulation
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    eval_time : float or None
        Time point to evaluate R. If None, use the end of simulation
    time_window : int
        Number of time points to average around eval_time
    
    Returns:
    --------
    R_value : float
        Average R value at the specified time
    """
    df = pd.read_csv(csv_file)
    
    if eval_time is None:
        # Use the end of simulation
        if len(df) <= time_window:
            return df['R_mean'].mean()
        else:
            return df['R_mean'].iloc[-time_window:].mean()
    else:
        # Find the closest time point to eval_time
        closest_idx = (df['time'] - eval_time).abs().idxmin()
        
        # Calculate window boundaries
        half_window = time_window // 2
        start_idx = max(0, closest_idx - half_window)
        end_idx = min(len(df) - 1, closest_idx + half_window)
        
        # Return the average R value within the window
        return df['R_mean'].iloc[start_idx:end_idx+1].mean()

def plot_R_vs_K(result_files, args):
    """Plot the order parameter R vs coupling strength K for different eps values"""
    # Extract data from all files
    data = []
    for file in result_files:
        params = extract_params_from_filename(file)
        R_value = calculate_R_at_time(file, args.eval_time, args.time_window)
        data.append({
            'K': params['K'],
            'eps': params['eps'],
            'attack_type': params['attack_type'],
            'start_time': params['start_time'],
            'end_time': params['end_time'],
            'is_random': params['is_random'],
            'R': R_value,
            'file': file
        })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df = df[(df['start_time'] == 0) & (df['end_time'] == -999.0)]
    
    # Get unique eps values
    eps_values = sorted(df['eps'].unique())
    
    # Set up the color map
    cmap = cm.coolwarm
    if len(eps_values) > 1:
        colors = [cmap(i / (len(eps_values) - 1)) for i in range(len(eps_values))]
    else:
        colors = [cmap(0.5)]
    
    # Create the plot
    plt.figure()
    plt.tick_params(labelsize=14)
    
    # For each eps value, plot both targeted and random perturbations (if available)
    for i, eps in enumerate(eps_values[::-1]):
        # Targeted perturbation
        subset_targeted = df[(df['eps'] == eps) & (~df['is_random'])]
        if not subset_targeted.empty:
            subset_targeted = subset_targeted.sort_values('K')
            plt.plot(subset_targeted['K'], subset_targeted['R'], 'o--', 
                    color=colors[len(eps_values) - i - 1], 
                    label=f'${eps}$',
                    linewidth=1,
                    markersize=8,
                    markeredgecolor='#333333',
                    markeredgewidth=0.5)
    
    # Add reference lines
    # plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    # plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    
    # Set plot labels and title
    plt.xlabel('Coupling Strength (K)', fontsize=20)
    plt.ylabel('Order Parameter (R)', fontsize=20)
    plt.title(f'(a) {args.network}', fontsize=20)
    # plt.title(f'(b) Brain', fontsize=20)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_time is not None:
        time = args.eval_time
    else:
        time = args.tmax
    
    if args.network in real_network_data:
        output_file = os.path.join(args.output_dir, f'R_vs_K_{args.network}_t{time}_{args.attack_type}_attack_interval{args.t_interval_values}.png')
    else:
        output_file = os.path.join(args.output_dir, f'R_vs_K_{args.network}_N{args.N_values}_kave{args.kave_values}_t{time}_{args.attack_type}_attack_interval{args.t_interval_values}.png')
    plt.legend(fontsize=10, title="$\epsilon$", title_fontsize=10)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def plot_R_vs_eps(result_files, args):
    """Plot the order parameter R vs perturbation strength eps for different K values"""
    # Extract data from all files
    data = []
    for file in result_files:
        params = extract_params_from_filename(file)
        R_value = calculate_R_at_time(file, args.eval_time, args.time_window)
        data.append({
            'K': params['K'],
            'eps': params['eps'],
            't_interval': params['t_interval'],
            'attack_type': params['attack_type'],
            'is_random': params['is_random'],
            'start_time': params['start_time'],
            'end_time': params['end_time'],
            'R': R_value,
            'file': file
        })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df = df[(df['start_time'] == 0) & (df['end_time'] == -999.0)]
    
    # Get unique t_interval values
    t_interval_values = sorted(df['t_interval'].unique())
    t_interval_values.remove(-999.0)
    
    # Create the plot
    plt.figure(figsize=(4.8, 4.8))
    plt.tick_params(labelsize=14)

    symbols = ['o--', 's--', '^--', 'v--', 'h--']
    # For each K value, plot both targeted and random perturbations (if available)
    for i, t_interval in enumerate(t_interval_values):
        # Targeted perturbation
        subset_targeted = df[((df['t_interval'] == t_interval) | (df['t_interval'] == -999)) & (~df['is_random'])]
        if not subset_targeted.empty:
            subset_targeted = subset_targeted.sort_values('eps')
            if min(subset_targeted['eps']) < 0.0:
                plt.plot(abs(subset_targeted['eps'][::-1]), subset_targeted['R'][::-1], symbols[i], 
                    label=f'Adversarial ($t_i={t_interval}$)',
                    color="black", 
                    linewidth=1,
                    markersize=8)
            else:
                plt.plot(subset_targeted['eps'], subset_targeted['R'], symbols[i], 
                    label=f'Adversarial ($t_i={t_interval}$)',
                    color="black", 
                    linewidth=1,
                    markersize=8)

        # Random perturbation
        subset_random = df[((df['t_interval'] == t_interval) | (df['t_interval'] == -999)) & ((df['is_random']) | (df['eps']==0.0))]
        if not subset_random.empty:
            subset_random = subset_random.sort_values('eps')
            if min(subset_random['eps']) < 0.0:
                plt.plot(abs(subset_random['eps'][::-1]), subset_random['R'][::-1], 'x--', 
                    color="black", 
                    label=f'Random',
                    linewidth=1,
                    markersize=8)
            else:
                plt.plot(subset_random['eps'], subset_random['R'], 'x--', 
                    color="black", 
                    label=f'Random',
                    linewidth=1,
                    markersize=8)

    
    # Set plot labels and title
    plt.title(f'{args.network}', fontsize=20)
    plt.xlabel('Perturbation Strength ($|\epsilon|$)', fontsize=20)
    plt.ylabel('Order Parameter (R)', fontsize=20)
    plt.ylim(-0.02, 1.02)
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_time is not None:
        time = args.eval_time
    else:
        time = args.tmax
    output_file = os.path.join(args.output_dir, f'R_vs_eps_{args.network}_N{args.N_values}_kave{args.kave_values}_K{args.K_values}_t{time}_{args.attack_type}_attack_interval{args.t_interval_values}.png')
    plt.legend(fontsize=10)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def plot_R_vs_N(result_files, args):
    """Plot the order parameter R vs perturbation strength eps for different K values"""
    # Extract data from all files
    data = []
    for file in result_files:
        params = extract_params_from_filename(file)
        R_value = calculate_R_at_time(file, args.eval_time, args.time_window)
        data.append({
            'N': params['N'],
            'K': params['K'],
            'eps': params['eps'],
            'attack_type': params['attack_type'],
            'is_random': params['is_random'],
            'start_time': params['start_time'],
            'end_time': params['end_time'],
            'R': R_value,
            'file': file
        })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df = df[(df['start_time'] == 0) & (df['end_time'] == -999.0)]

    # Get unique eps values
    eps_values = sorted(df['eps'].unique())
    
    # Create the plot
    plt.figure(figsize=(4.8, 4.8))
    plt.tick_params(labelsize=14)

    symbols = ['o--', 's--', '^--', 'v--', 'h--']
    # For each K value, plot both targeted and random perturbations (if available)
    if min(eps_values) < 0.0:
        eps_values = eps_values[::-1]
        
    for i, eps in enumerate(eps_values):
        # Targeted perturbation
        subset_targeted = df[(df['eps'] == eps) & (~df['is_random'])]
        if not subset_targeted.empty:
            subset_targeted = subset_targeted.sort_values('N')
            plt.plot(subset_targeted['N'], subset_targeted['R'], symbols[i], 
                    label=f'Adversarial ($\epsilon={eps}$)',
                    color="black",
                    linewidth=1,
                    markersize=8)
        
        # Random perturbation
        subset_random = df[(df['eps'] == eps) & (df['is_random'])]
        if not subset_random.empty:
            subset_random = subset_random.sort_values('N')
            plt.plot(subset_random['N'], subset_random['R'], 'x--', 
                    color="black", 
                    label=f'Random',
                    linewidth=1,
                    markersize=8)
    
    # Set plot labels and title
    plt.title(f'{args.network}', fontsize=20)
    plt.xlabel('Network Size (N)', fontsize=20)
    plt.ylabel('Order Parameter (R)', fontsize=20)
    plt.ylim(-0.02, 1.02)
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_time is not None:
        time = args.eval_time
    else:
        time = args.tmax
    output_file = os.path.join(args.output_dir, f'R_vs_N_{args.network}_kave{args.kave_values}_K{args.K_values}_eps{args.eps_values}_t{time}_{args.attack_type}_attack_interval{args.t_interval_values}.png')
    plt.legend(fontsize=10)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def main():
    args = parse_args()
    print(f"Searching for result files in: {args.results_dir}")
    
    # Find all matching result files
    result_files = find_result_files(args)
    
    if not result_files:
        print("No matching result files found. Check your parameters.")
        return
    
    print(f"Found {len(result_files)} result files.")
    
    # Generate the appropriate plot based on the plot_type argument
    if args.plot_type == 'K_vs_R' or args.plot_type.lower() == 'k_vs_r':
        plot_R_vs_K(result_files, args)
    elif args.plot_type == 'eps_vs_R' or args.plot_type.lower() == 'eps_vs_r':
        plot_R_vs_eps(result_files, args)
    elif args.plot_type == 'N_vs_R' or args.plot_type.lower() == 'n_vs_r':
        plot_R_vs_N(result_files, args)
    else:
        print(f"Unknown plot type: {args.plot_type}.")
    
    print("Plotting complete!")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import re
from scipy.stats import pearsonr

def extract_round_number(filename):
    """Extract round number from filename."""
    match = re.search(r'round_(\d+)', filename)
    return int(match.group(1)) if match else -1

def plot_final_result(signals_dict, output_path):
    """Plot only the final result using the highest round number.
    
    Args:
        signals_dict: Dictionary with signal names as keys and signal data as values
        output_path: Path to save the plot
    """
    # Convert all signals to flat arrays
    processed_signals = {
        name: np.array(signal)[:5].flatten() 
        for name, signal in signals_dict.items()
    }
    
    # Find the highest round number
    rounds = set()
    for name in processed_signals.keys():
        round_num = extract_round_number(name)
        if round_num > 0:
            rounds.add(round_num)
    max_round = max(rounds)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Get signals for final result
    input_signal = processed_signals.get(f'input_round_{max_round}.json')
    expected_signal = processed_signals.get(f'expected_output_round_{max_round}.json')
    predicted_signal = processed_signals.get(f'output_round_{max_round}.json')
    
    # Create time axis
    signal_length = len(input_signal)
    time = np.arange(signal_length)
    
    # Plot PPG input signal
    plt.subplot(2, 1, 1)
    plt.plot(time, input_signal, linewidth=1.5, color='#2ecc71', 
            label='PPG (Entrada)', linestyle='-')
    plt.title('Sinal PPG de Entrada', fontsize=12, pad=10)
    plt.xlabel('Amostras', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # Plot ECG signals comparison
    plt.subplot(2, 1, 2)
    plt.plot(time, expected_signal, linewidth=1.5, color='#3498db', 
            label='ECG Esperado', linestyle='-')
    plt.plot(time, predicted_signal, linewidth=1.5, color='#e74c3c', 
            label='ECG Predito', linestyle='-')
    plt.title(f'Comparação ECG - Round {max_round}', fontsize=12, pad=10)
    plt.xlabel('Amostras', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_signals(signals_dict, output_path):
    """Plot all signals from the sample directory.
    
    Args:
        signals_dict: Dictionary with signal names as keys and signal data as values
        output_path: Path to save the plot
    """
    # Convert all signals to flat arrays
    processed_signals = {
        name: np.array(signal)[:5].flatten() 
        for name, signal in signals_dict.items()
    }
    
    # Group signals by round
    rounds = set()
    for name in processed_signals.keys():
        round_num = extract_round_number(name)
        if round_num > 0:
            rounds.add(round_num)
    rounds = sorted(list(rounds))
    
    # Calculate number of plots needed (one plot per round plus input)
    total_plots = len(rounds) + 1
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 4 * total_plots))
    
    # Plot input signal first
    plt.subplot(total_plots, 1, 1)
    input_signal = processed_signals.get(f'input_round_{rounds[0]}.json')
    time = np.arange(len(input_signal))
    plt.plot(time, input_signal, linewidth=1.5, color='#2ecc71',
            label='PPG (Entrada)', linestyle='-')
    plt.title('Sinal PPG de Entrada', fontsize=12, pad=10)
    plt.xlabel('Amostras', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # Plot each round's ECG comparison
    for idx, round_num in enumerate(rounds, 2):
        plt.subplot(total_plots, 1, idx)
        
        # Get signals for this round
        expected_signal = processed_signals.get(f'expected_output_round_{round_num}.json')
        predicted_signal = processed_signals.get(f'output_round_{round_num}.json')
        
        # Create time axis
        time = np.arange(len(expected_signal))
        
        # Plot signals
        plt.plot(time, expected_signal, linewidth=1.5, color='#3498db', 
                label='ECG Esperado', linestyle='-')
        plt.plot(time, predicted_signal, linewidth=1.5, color='#e74c3c', 
                label='ECG Predito', linestyle='-')
        
        plt.title(f'Comparação ECG - Round {round_num}', fontsize=12, pad=10)
        plt.xlabel('Amostras', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(h_pad=1.0)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def calculate_signal_metrics(expected_signal, predicted_signal):
    """Calculate metrics between expected and predicted signals.
    
    Args:
        expected_signal: Array containing the expected ECG signal
        predicted_signal: Array containing the predicted ECG signal
        
    Returns:
        Dictionary containing correlation metrics
    """
    corr, _ = pearsonr(expected_signal, predicted_signal)
    
    return {
        'correlation': corr
    }

def process_client_data(client_dir: Path, output_dir: Path):
    """Process data for a single client."""
    # Create client output directory
    client_output_dir = output_dir / client_dir.name
    client_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save sample signals if they exist
    sample_dir = client_dir / 'sample'
    metrics_by_round = {}
    
    if sample_dir.exists():
        # Load all JSON files from sample directory
        signals = {}
        for json_file in sorted(sample_dir.glob('*.json')):
            with open(json_file, 'r') as f:
                signals[json_file.name] = json.load(f)
        
        # Calculate metrics for each round
        rounds = set()
        for name in signals.keys():
            round_num = extract_round_number(name)
            if round_num > 0:
                rounds.add(round_num)
        
        for round_num in sorted(rounds):
            expected_signal = np.array(signals[f'expected_output_round_{round_num}.json']).flatten()
            predicted_signal = np.array(signals[f'output_round_{round_num}.json']).flatten()
            
            metrics = calculate_signal_metrics(expected_signal, predicted_signal)
            metrics_by_round[round_num] = metrics
            
        # Plot signals if any were found
        if signals:
            # Plot all rounds progression
            plot_signals(signals, client_output_dir / 'sample_signals.pdf')
            # Plot final result
            plot_final_result(signals, client_output_dir / 'final_result.pdf')
            
        # Save metrics to JSON
        if metrics_by_round:
            metrics_path = client_output_dir / 'signal_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_by_round, f, indent=4)
    
    return metrics_by_round

def process_experiment(experiment_path: Path, output_path: Path):
    """Process a single experiment folder and generate plots."""
    # Read the global history file
    history_df = pd.read_csv(experiment_path / 'server_metrics.csv')
    
    # Create output directory if it doesn't exist
    output_dir = output_path / experiment_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each client's data and collect metrics
    all_client_metrics = {}
    for client_dir in experiment_path.glob('client_*'):
        if client_dir.is_dir():
            client_metrics = process_client_data(client_dir, output_dir)
            all_client_metrics[client_dir.name] = client_metrics
    
    # Calculate average and std of metrics across clients for each round
    rounds = set()
    for client_metrics in all_client_metrics.values():
        rounds.update(client_metrics.keys())
    
    aggregated_metrics = {}
    for round_num in sorted(rounds):
        round_correlations = [
            metrics[round_num]['correlation']
            for client, metrics in all_client_metrics.items()
            if round_num in metrics
        ]
        
        aggregated_metrics[round_num] = {
            'mean_correlation': float(np.mean(round_correlations)),
            'std_correlation': float(np.std(round_correlations)),
            'client_correlations': round_correlations
        }
    
    # Save aggregated metrics
    metrics_path = output_dir / 'aggregated_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'by_round': aggregated_metrics,
            'by_client': all_client_metrics
        }, f, indent=4)
    
    # Set style for better looking plots
    plt.style.use('ggplot')
    
    # Plot training history with correlation metrics
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history_df['loss'], label='Perda Global', linewidth=2)
    plt.title('Função de Perda', fontsize=12, pad=10)
    plt.xlabel('Round', fontsize=10)
    plt.ylabel('Perda (MSE)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE
    plt.subplot(1, 3, 2)
    plt.plot(history_df['rmse'], label='RMSE Global', linewidth=2)
    plt.title('Erro Quadrático Médio', fontsize=12, pad=10)
    plt.xlabel('Round', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot correlation metrics
    plt.subplot(1, 3, 3)
    rounds = sorted(aggregated_metrics.keys())
    means = [aggregated_metrics[r]['mean_correlation'] for r in rounds]
    stds = [aggregated_metrics[r]['std_correlation'] for r in rounds]
    
    plt.errorbar(rounds, means, yerr=stds, fmt='-o', capsize=5,
                label='Correlação Média entre Clientes', linewidth=2)
    plt.title('Correlação de Pearson', fontsize=12, pad=10)
    plt.xlabel('Round', fontsize=10)
    plt.ylabel('Correlação', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Save processed history as CSV
    history_df.to_csv(output_dir / 'processed_history.csv', index=False)

def main():
    # Setup paths
    results_dir = Path('results/fedavg')
    output_dir = Path('processed_results/fedavg')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each experiment folder
    for experiment_folder in results_dir.iterdir():
        if experiment_folder.is_dir():
            print(f"Processing experiment: {experiment_folder.name}")
            process_experiment(experiment_folder, output_dir)
            print(f"Finished processing: {experiment_folder.name}")

if __name__ == "__main__":
    main()

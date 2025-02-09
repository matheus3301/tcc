import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import re

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
        name: np.array(signal).flatten() 
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
        name: np.array(signal).flatten() 
        for name, signal in signals_dict.items()
    }
    
    # Group signals by round
    rounds = set()
    for name in processed_signals.keys():
        round_num = extract_round_number(name)
        if round_num > 0:
            rounds.add(round_num)
    rounds = sorted(list(rounds))
    
    # Calculate number of plots needed (one plot per round)
    total_plots = len(rounds)
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 4 * total_plots))
    
    # Plot each round's signals
    for idx, round_num in enumerate(rounds, 1):
        plt.subplot(total_plots, 1, idx)
        
        # Get signals for this round
        input_signal = processed_signals.get(f'input_round_{round_num}.json')
        expected_signal = processed_signals.get(f'expected_output_round_{round_num}.json')
        predicted_signal = processed_signals.get(f'output_round_{round_num}.json')
        
        # Create time axis
        signal_length = len(input_signal)
        time = np.arange(signal_length)
        
        # Plot signals
        plt.plot(time, input_signal, linewidth=1.5, color='#2ecc71', 
                label='PPG (Entrada)', linestyle='-')
        plt.plot(time, expected_signal, linewidth=1.5, color='#3498db', 
                label='ECG Esperado', linestyle='-')
        plt.plot(time, predicted_signal, linewidth=1.5, color='#e74c3c', 
                label='ECG Predito', linestyle='-')
        
        plt.title(f'Comparação de Sinais - Round {round_num}', fontsize=12, pad=10)
        plt.xlabel('Amostras', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(h_pad=1.0)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def process_client_data(client_dir: Path, output_dir: Path):
    """Process data for a single client."""
    # Create client output directory
    client_output_dir = output_dir / client_dir.name
    client_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save sample signals if they exist
    sample_dir = client_dir / 'sample'
    if sample_dir.exists():
        # Load all JSON files from sample directory
        signals = {}
        for json_file in sorted(sample_dir.glob('*.json')):
            with open(json_file, 'r') as f:
                signals[json_file.name] = json.load(f)
            
        # Plot signals if any were found
        if signals:
            # Plot all rounds progression
            plot_signals(signals, client_output_dir / 'sample_signals.pdf')
            # Plot final result
            plot_final_result(signals, client_output_dir / 'final_result.pdf')

def process_experiment(experiment_path: Path, output_path: Path):
    """Process a single experiment folder and generate plots."""
    # Read the global history file
    history_df = pd.read_csv(experiment_path / 'history.csv')
    
    # Create output directory if it doesn't exist
    output_dir = output_path / experiment_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each client's data
    for client_dir in experiment_path.glob('client_*'):
        if client_dir.is_dir():
            process_client_data(client_dir, output_dir)
    
    # Set style for better looking plots
    plt.style.use('ggplot')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Perda Global', linewidth=2)
    plt.title('Função de Perda', fontsize=12, pad=10)
    plt.xlabel('Round', fontsize=10)
    plt.ylabel('Perda (MSE)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE
    plt.subplot(1, 2, 2)
    plt.plot(history_df['rmse'], label='RMSE Global', linewidth=2)
    plt.title('Erro Quadrático Médio', fontsize=12, pad=10)
    plt.xlabel('Round', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PDF
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

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

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
    
    # Create time axis
    signal_length = len(next(iter(processed_signals.values())))
    time = np.arange(signal_length)
    
    # Organize signals by type
    ppg_signal = processed_signals.get('input.json')
    expected_ecg = processed_signals.get('expected_output.json')
    predicted_signals = {k: v for k, v in processed_signals.items() 
                        if k not in ['input.json', 'expected_output.json']}
    
    # Calculate number of comparison plots needed
    n_comparisons = len(predicted_signals)
    total_plots = n_comparisons + 1  # +1 for PPG plot
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 4 * total_plots))
    
    # Find signal boundaries (assuming all signals have same length)
    n_signals_per_file = len(ppg_signal) // signal_length if ppg_signal is not None else 1
    signal_boundaries = [i * signal_length // n_signals_per_file for i in range(1, n_signals_per_file)]
    
    # Plot PPG signal first
    if ppg_signal is not None:
        plt.subplot(total_plots, 1, 1)
        plt.plot(time, ppg_signal, linewidth=1.5, color='#2ecc71')
        plt.title('Sinal PPG de Entrada', fontsize=12, pad=10)
        plt.xlabel('Amostras', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines to separate signals
        for boundary in signal_boundaries:
            plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.8)
            # Adiciona numeração dos sinais
            plt.text(boundary - signal_length//(2*n_signals_per_file), 
                    plt.ylim()[0], 
                    f'Sinal {boundary//(signal_length//n_signals_per_file)}', 
                    horizontalalignment='center')
    
    # Plot each predicted signal with its corresponding expected signal
    for idx, (name, pred_signal) in enumerate(predicted_signals.items(), 2):
        plt.subplot(total_plots, 1, idx)
        
        # Plot expected signal
        if expected_ecg is not None:
            plt.plot(time, expected_ecg, linewidth=1.5, color='#3498db', 
                    label='ECG Esperado', linestyle='-')
        
        # Plot predicted signal
        plt.plot(time, pred_signal, linewidth=1.5, color='#e74c3c', 
                label='ECG Predito', linestyle='-')
        
        plt.title(f'Comparação ECG - {name.replace(".json", "").replace("output", "Amostra")}', 
                 fontsize=12, pad=10)
        plt.xlabel('Amostras', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # Add vertical lines to separate signals
        for boundary in signal_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            # Adiciona numeração dos sinais
            plt.text(boundary - signal_length//(2*n_signals_per_file), 
                    plt.ylim()[0], 
                    f'Sinal {boundary//(signal_length//n_signals_per_file)}', 
                    horizontalalignment='center')
    
    # Adjust layout to prevent overlap
    plt.tight_layout(h_pad=1.0)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def calculate_signal_metrics(expected_signal, predicted_signal, n_samples):
    """Calculate metrics between expected and predicted signals.
    
    Args:
        expected_signal: Array containing the expected ECG signals
        predicted_signal: Array containing the predicted ECG signals
        n_samples: Number of individual signals in the arrays
        
    Returns:
        Dictionary containing correlation metrics for each signal pair
    """
    # Calculate length of each individual signal
    total_length = len(expected_signal)
    sample_length = total_length // n_samples
    
    # Split signals into individual samples
    expected_samples = [expected_signal[i*sample_length:(i+1)*sample_length] for i in range(n_samples)]
    predicted_samples = [predicted_signal[i*sample_length:(i+1)*sample_length] for i in range(n_samples)]
    
    # Calculate correlation for each pair
    pair_correlations = []
    for i in range(n_samples):
        corr, _ = pearsonr(expected_samples[i], predicted_samples[i])
        pair_correlations.append(corr)
    
    metrics = {
        'individual_correlations': pair_correlations,
        'mean_correlation': np.mean(pair_correlations),
        'std_correlation': np.std(pair_correlations)
    }
    
    return metrics

def process_experiment(experiment_path: Path, output_path: Path):
    """Process a single experiment folder and generate plots."""
    # Read the history file
    history_df = pd.read_csv(experiment_path / 'history.csv')
    
    # Create output directory if it doesn't exist
    output_dir = output_path / experiment_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save sample signals if they exist
    sample_dir = experiment_path / 'sample'
    if sample_dir.exists():
        # Load all JSON files from sample directory
        signals = {}
        for json_file in sorted(sample_dir.glob('*.json')):
            with open(json_file, 'r') as f:
                signals[json_file.name] = json.load(f)
        
        # Calculate metrics if signals exist
        if signals:
            expected_ecg = np.array(signals.get('expected_output.json')).flatten()
            metrics = {}
            
            for name, signal in signals.items():
                if name not in ['input.json', 'expected_output.json']:
                    predicted_signal = np.array(signal).flatten()
                    metrics[name] = calculate_signal_metrics(expected_ecg, predicted_signal, n_samples=10)
                    
                    # Print metrics for this prediction
                    print(f"\nMetrics for {name}:")
                    for i, corr in enumerate(metrics[name]['individual_correlations']):
                        print(f"ECG pair {i+1}: Correlation = {corr:.4f}")
                    print(f"Mean correlation: {metrics[name]['mean_correlation']:.4f}")
                    print(f"Std correlation: {metrics[name]['std_correlation']:.4f}")
            
            # Save metrics to JSON
            metrics_path = output_dir / 'signal_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot signals
            plot_signals(signals, output_dir / 'sample_signals.pdf')
    
    # Set style for better looking plots
    plt.style.use('ggplot')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Perda no Treino', linewidth=2)
    plt.plot(history_df['val_loss'], label='Perda na Validação', linewidth=2)
    plt.title('Função de Perda', fontsize=12, pad=10)
    plt.xlabel('Época', fontsize=10)
    plt.ylabel('Perda (MSE)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE
    plt.subplot(1, 2, 2)
    plt.plot(history_df['rmse'], label='RMSE no Treino', linewidth=2)
    plt.plot(history_df['val_rmse'], label='RMSE na Validação', linewidth=2)
    plt.title('Erro Quadrático Médio', fontsize=12, pad=10)
    plt.xlabel('Época', fontsize=10)
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
    results_dir = Path('results/centralized')
    output_dir = Path('processed_results/centralized')
    
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

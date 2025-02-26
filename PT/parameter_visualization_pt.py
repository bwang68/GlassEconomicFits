import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple

def plot_parameter_changes(parameter_history, objective_history=None):
    """
    Plot the changes in parameters during model fitting
    parameter_history: dict with keys as parameter names and values as lists of parameter values
    objective_history: list of objective function values
    """
    parameters = ['a', 'b', 'g', 'l', 'tw', 'epsilon_gains', 'epsilon_losses']
    n_plots = len(parameters) + 1 if objective_history is not None else len(parameters)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots))
    fig.suptitle('Parameter Changes and Optimization Progress')
    
    # Plot parameters
    for idx, param in enumerate(parameters):
        if param in parameter_history:
            axes[idx].plot(parameter_history[param], marker='o')
            axes[idx].set_ylabel(param)
            axes[idx].grid(True)
    
    # Plot objective function if provided
    if objective_history is not None:
        idx = len(parameters)
        axes[idx].plot(objective_history, 'b-', alpha=0.6, label='Objective Value')
        axes[idx].plot(objective_history, 'r.', alpha=0.4, label='Iterations')
        
        # Add trend line
        z = np.polyfit(range(len(objective_history)), objective_history, 1)
        p = np.poly1d(z)
        axes[idx].plot(range(len(objective_history)), 
                      p(range(len(objective_history))), 
                      "r--", alpha=0.8, label='Trend')
        
        axes[idx].set_ylabel('MSE')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders

def track_parameters(model):
    """
    Initialize a dictionary to track parameter changes
    """
    return {
        'a': [],
        'b': [],
        'g': [],
        'l': [],
        'tw': [],
        'epsilon_gains': [],
        'epsilon_losses': []
    }

def update_parameter_history(parameter_history, model):
    """
    Update the parameter history with current model parameters
    """
    parameter_history['a'].append(model.a)
    parameter_history['b'].append(model.b)
    parameter_history['g'].append(model.g)
    parameter_history['l'].append(model.l)
    parameter_history['tw'].append(model.tw)
    parameter_history['epsilon_gains'].append(model.epsilon_gains)
    parameter_history['epsilon_losses'].append(model.epsilon_losses)
    
    return parameter_history

def plot_predictions_vs_actuals(model, participant: int, params) -> None:
    """Plot predicted vs actual sales for a participant"""
    predictions = model.predict_sales(participant, params)
    actuals = [model.info_return(day, participant)[0] for day in range(68)]
    
    plt.figure(figsize=(12, 8))
    
    # Plot time series
    plt.subplot(2, 1, 1)
    plt.plot(range(68), actuals, 'b-', label='Actual Sales', alpha=0.6)
    plt.plot(range(68), predictions, 'r--', label='Predicted Sales', alpha=0.6)
    plt.title(f'Participant {participant}: Sales Over Time')
    plt.xlabel('Day')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot scatter
    plt.subplot(2, 1, 2)
    plt.scatter(actuals, predictions, alpha=0.5)
    max_val = max(max(actuals), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    plt.title('Predicted vs Actual Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(actuals, predictions)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
             transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    input(f"Press Enter to continue to next plot...")  # Wait for user input
    plt.close()

def visualize_and_print_results(model, results: dict, output_dir: str):
    """Visualize results and print summaries for all participants"""
    plt.ion()  # Turn on interactive mode
    plt.close('all')  # Clear any existing plots
    
    # ...rest of your existing visualization code...

def visualize_participant_results(model, results: dict):
    """Visualize results for all participants"""
    for participant, result in results.items():
        if 'error' in result and isinstance(result['error'], (int, float)):
            try:
                # Print results
                print(f"\nParticipant {participant}:")
                for key, value in result.items():
                    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                
                # Create Parameters object
                params = Parameters(
                    a=result['alpha'],
                    b=result['beta'],
                    l=result['lambda'],
                    g=result['gamma'],
                    tw=float(result['time_window'])
                )
                
                # Show plots
                plot_predictions_vs_actuals(model, participant, params)
                
            except Exception as e:
                print(f"\nError visualizing participant {participant}: {e}")

# ...rest of existing code...

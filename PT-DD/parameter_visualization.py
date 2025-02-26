import matplotlib.pyplot as plt
import numpy as np

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
    plt.show(block=True)

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

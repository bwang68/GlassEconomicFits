import pandas as pd
import numpy as np
from functools import lru_cache
from tqdm import tqdm
from dataclasses import dataclass
from scipy.optimize import least_squares, basinhopping
import sys
from parameter_visualization_pt_tw import track_parameters, plot_parameter_changes
import matplotlib.pyplot as plt

CACHE_SIZE = 10000
DATA_DIR = '.'

@dataclass(frozen=True)
class Parameters:
    a: float = 1.0
    b: float = 1.0
    g: float = 1.0
    l: float = 1.0
    tw: int = 68
    epsilon_gains: float = 0.0
    epsilon_losses: float = 0.0

def load_dataset() -> pd.DataFrame:
    try:
        data = pd.read_csv("/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv")
        print(f"Successfully loaded dataset from CSV")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

class PT_Model:
    def __init__(self):
        self.data = None
        self.best_fits = {}
        self.parameter_history = {}
        self.objective_history = {}  # Add this to store objective values
        self.subject_errors = {}  # Add this to store errors for each subject

    @lru_cache(maxsize=CACHE_SIZE)
    def probability_weighting_function(self, probability: float, gamma: float) -> float:
        """Applies the probability weighting function."""
        return probability ** gamma

    @lru_cache(maxsize=CACHE_SIZE)
    def gain(self, amount: int, price: int, j: int, a: float, g: float) -> float:
        


        price_diff = j - price
        if price_diff <= 0:
            return 0  # No gain if future price is less than or equal to the current price
        inner_bracket1 = amount * (price_diff) ** a
        return inner_bracket1 * self.probability_weighting_function(1 / j, g)

    @lru_cache(maxsize=CACHE_SIZE)
    def loss(self, amount: int, price: int, j: int, b: float, l: float, g: float) -> float:
        price_diff = price - j
        if price_diff <= 0:
            return 0  # No loss if future price is greater than or equal to the current price
        inner_bracket2 = amount * (price_diff) ** b
        return inner_bracket2 * self.probability_weighting_function(1 / j, g) * l

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_gain(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """Expected value calculation for gains."""
        ev = 0.0
        max_price = 15
        for future_day in range(day + 1, 69):
            time_diff = future_day - day
            
            for future_price in range(price + 1, max_price + 1):
                price_diff = future_price - price
                gain_value = self.gain(amount, price, future_price, fit.a, fit.g)
                ev += gain_value
        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_loss(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """Expected value calculation for losses."""
        ev = 0.0
        for future_day in range(day + 1, 69):
            time_diff = future_day - day
            for future_price in range(1, price):
                price_diff = price - future_price
                loss_value = self.loss(amount, price, future_price, fit.b, fit.l, fit.g)
                ev += loss_value
        return ev

    def error_of_fit(self, subject: int, fit: Parameters) -> float:
        """Calculates the error for a given set of parameters."""
        subject_data = self.data[self.data['Subject'] == subject]
        stored_cols = [col for col in self.data.columns if col.startswith('Stored')]
        sold_cols = [col for col in self.data.columns if col.startswith('Sold')]
        price_cols = [col for col in self.data.columns if col.startswith('Price')]
        stored = subject_data[stored_cols].values.flatten()
        sold = subject_data[sold_cols].values.flatten()
        prices = subject_data[price_cols].values.flatten()
        days = np.arange(1, len(stored) + 1)

        predicted_sold = self.predict_sales(subject, days, stored, prices, sold, fit)
        error = np.sum((sold - predicted_sold) ** 2)  # Mean Squared Error
        return error

    def predict_sales(self, subject: int, days: np.ndarray, stored: np.ndarray, prices: np.ndarray, actual_sold: np.ndarray, params: Parameters) -> np.ndarray:
        predicted_sold = np.zeros_like(stored)

        for idx, day in enumerate(days):
            amount = stored[idx]
            price = prices[idx]

            if amount == 0:  # Avoid unnecessary calculations for zero amount
                predicted_sold[idx] = 0
                continue

            ev_gain = self.expected_value_gain(day, price, amount, params)
            ev_loss = self.expected_value_loss(day, price, amount, params)

            if ev_gain >= abs(ev_loss):
                predicted_sold[idx] = amount
            else:
                predicted_sold[idx] = 0

        return predicted_sold

    def fit_one_subject(self, subject: int, start_fit: Parameters) -> None:
        """Fits the model for one subject using Basin-Hopping algorithm."""
        # Initialize parameter and objective history
        self.parameter_history[subject] = {
            'a': [], 'b': [], 'g': [], 'l': [],
            'tw': [], 'epsilon_gains': [], 'epsilon_losses': []
        }
        self.objective_history[subject] = []

        # Get the actual sales data for the subject
        subject_data = self.data[self.data['Subject'] == subject]
        stored_cols = [col for col in self.data.columns if col.startswith('Stored')]
        sold_cols = [col for col in self.data.columns if col.startswith('Sold')]
        price_cols = [col for col in self.data.columns if col.startswith('Price')]
        
        stored = subject_data[stored_cols].values.flatten()
        actual_sold = subject_data[sold_cols].values.flatten()
        prices = subject_data[price_cols].values.flatten()
        days = np.arange(1, len(stored) + 1)

        def objective_function(params):
            """Calculate objective function for Basin-Hopping."""
            # Bound parameters to their valid ranges
            bounded_params = np.clip(params, [0.01, 0.01, 0.01, 1.0, 0.0, 0.0], 
                                          [1.0, 1.0, 3.0, 10.0, 1.0, 1.0])
            
            fit = Parameters(a=bounded_params[0], 
                           b=bounded_params[1], 
                           g=bounded_params[2], 
                           l=bounded_params[3],
                           tw=start_fit.tw, 
                           epsilon_gains=bounded_params[4], 
                           epsilon_losses=bounded_params[5])
            
            # Update parameter history
            self.parameter_history[subject]['a'].append(fit.a)
            self.parameter_history[subject]['b'].append(fit.b)
            self.parameter_history[subject]['g'].append(fit.g)
            self.parameter_history[subject]['l'].append(fit.l)
            self.parameter_history[subject]['tw'].append(fit.tw)
            self.parameter_history[subject]['epsilon_gains'].append(fit.epsilon_gains)
            self.parameter_history[subject]['epsilon_losses'].append(fit.epsilon_losses)
            
            predicted = self.predict_sales(subject, days, stored, prices, actual_sold, fit)
            obj_value = np.sum((actual_sold - predicted) ** 2)  # MSE
            
            # Store objective value
            self.objective_history[subject].append(obj_value)
            
            return obj_value

        # Initial parameter guess
        x0 = np.array([
            start_fit.a,
            start_fit.b,
            start_fit.g,
            start_fit.l,
            start_fit.epsilon_gains,
            start_fit.epsilon_losses
        ])

        # Define bounds for the parameters
        bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 3.0), 
                 (1.0, 10.0), (0.0, 1.0), (0.0, 1.0)]

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds
        }

        try:
            # Use Basin-Hopping algorithm with increased iterations
            result = basinhopping(
                objective_function,
                x0,
                niter=1000,          # Increased from 100 to 500 iterations
                T=1.0,              # Temperature parameter
                stepsize=0.05,      # Step size for random displacement
                minimizer_kwargs=minimizer_kwargs,
                interval=50,        # Increased interval for stepsize adjustment
                niter_success=20,   # Increased from 10 to 20 successful iterations needed
                disp=True,          # Display progress
                seed=42             # Add seed for reproducibility
            )

            # Save the best-fit parameters
            best_fit = Parameters(
                a=result.x[0],
                b=result.x[1],
                g=result.x[2],
                l=result.x[3],
                tw=start_fit.tw,
                epsilon_gains=result.x[4],
                epsilon_losses=result.x[5]
            )
            
            self.best_fits[subject] = best_fit
            print(f"\nSubject {subject} best-fit parameters: {best_fit}\n")
            print(f"Optimization success: {result.success}")
            print(f"Global minimum found: {result.lowest_optimization_result.fun}")
            print(f"Number of iterations: {result.nit}")
            
            # Plot parameter changes and optimization progress
            # plot_parameter_changes(self.parameter_history[subject], 
            #                    self.objective_history[subject])
            
            # Print MSE from error_of_fit
            final_mse = self.error_of_fit(subject, best_fit)
            print(f"\nFinal Mean Squared Error for Subject {subject}: {final_mse:.4f}")
            self.subject_errors[subject] = final_mse  # Store the error
            
        except Exception as e:
            print(f"Optimization failed for subject {subject}: {str(e)}")
            return None

    def fit_all_subjects(self, start_fit: Parameters):
        """Fit the model for all subjects."""
        for subject in self.data['Subject'].unique():
            print(f"Fitting subject {subject}")
            self.fit_one_subject(subject, start_fit)
        
        # Calculate mean error
        mean_error = np.mean(list(self.subject_errors.values()))
        
        # Write errors to file
        with open('model_errors.txt', 'w') as f:
            f.write("Individual Subject Errors:\n")
            for subject, error in self.subject_errors.items():
                f.write(f"Subject {subject}: {error:.4f}\n")
            f.write(f"\nMean Error Across All Subjects: {mean_error:.4f}\n")
            
        print(f"\nMean Error Across All Subjects: {mean_error:.4f}")
        print("Errors have been saved to model_errors.txt")

def main(version: str):
    model = PT_Model()
    data = load_dataset()
    model.data = data
    start_fit = Parameters(a=0.5, b=0.5, g=1.0, l=2.25, epsilon_gains=0.5, epsilon_losses=0.5)
    model.fit_all_subjects(start_fit)

if __name__ == '__main__':
    version = "tw_dd_v5"
    main(version=version)



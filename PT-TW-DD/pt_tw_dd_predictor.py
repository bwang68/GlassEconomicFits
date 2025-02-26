import pandas as pd
import numpy as np
from functools import lru_cache
from tqdm import tqdm
from dataclasses import dataclass
from scipy.optimize import differential_evolution
import sys
from datetime import datetime
from parameter_visualization import plot_parameter_changes  # Add this import

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
        print(f"Successfully loaded data")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

class PT_TW_DDModel:
    def __init__(self):
        self.data = None
        self.best_fits = {}
        self.subject_errors = {}
        self.parameter_history = None
        self.objective_history = None

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
    def delayed_discounting(self, time_diff: int, epsilon: float) -> float:
        if time_diff < 0:
            return 1.0
        return 1 / (1 + epsilon * time_diff)

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_gain(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """Expected value calculation for gains."""
        ev = 0.0
        max_price = 15
        for future_day in range(day + 1, day + fit.tw + 1):
            time_diff = future_day - day
            discount_factor = self.delayed_discounting(time_diff, fit.epsilon_gains)
            for future_price in range(price + 1, max_price + 1):
                price_diff = future_price - price
                gain_value = self.gain(amount, price, future_price, fit.a, fit.g)
                ev += gain_value * discount_factor
        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_loss(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """Expected value calculation for losses."""
        ev = 0.0
        for future_day in range(day + 1, day + fit.tw + 1):
            time_diff = future_day - day
            discount_factor = self.delayed_discounting(time_diff, fit.epsilon_losses)
            for future_price in range(1, price):
                price_diff = price - future_price
                loss_value = self.loss(amount, price, future_price, fit.b, fit.l, fit.g)
                ev += loss_value * discount_factor
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

    def fit_one_subject(self, subject: int, start_fit: Parameters) -> float:
        """Fits the model for one subject using differential evolution."""
        bounds = [(0.01, 1.0),  # 'a' (less than 1)
                  (0.01, 1.0),  # 'b' (less than 1)
                  (0.01, 3.0),  # 'g' (bounded based on typical range)
                  (1.0, 10.0),  # 'l' (greater than 1)
                  (0.0, 1.0),   # epsilon_gains
                  (0.0, 1.0)]   # epsilon_losses

        # Initialize parameter history
        self.parameter_history = {
            'a': [], 'b': [], 'g': [], 'l': [],
            'tw': [], 'epsilon_gains': [], 'epsilon_losses': []
        }
        self.objective_history = []

        def objective(params):
            fit = Parameters(a=params[0], b=params[1], g=params[2], l=params[3], 
                           tw=start_fit.tw, epsilon_gains=params[4], epsilon_losses=params[5])
            error = self.error_of_fit(subject, fit)
            
            # Record parameter values and objective
            self.parameter_history['a'].append(params[0])
            self.parameter_history['b'].append(params[1])
            self.parameter_history['g'].append(params[2])
            self.parameter_history['l'].append(params[3])
            self.parameter_history['tw'].append(start_fit.tw)
            self.parameter_history['epsilon_gains'].append(params[4])
            self.parameter_history['epsilon_losses'].append(params[5])
            self.objective_history.append(error)
            
            return error

        result = differential_evolution(objective, bounds)

        # Save the best-fit parameters
        best_fit = Parameters(a=result.x[0], b=result.x[1], g=result.x[2], l=result.x[3],
                            tw=start_fit.tw, epsilon_gains=result.x[4], epsilon_losses=result.x[5])
        self.best_fits[subject] = best_fit
        final_error = result.fun
        self.subject_errors[subject] = final_error

        # Plot parameter changes
        plot_parameter_changes(self.parameter_history, self.objective_history)
        
        print(f"\nSubject {subject} best-fit parameters: {best_fit}")
        print(f"Subject {subject} final error: {final_error}\n")
        return final_error

    def fit_all_subjects(self, start_fit: Parameters):
        """Fit the model for all subjects."""
        errors = []
        for subject in self.data['Subject'].unique():
            print(f"Fitting subject {subject}")
            error = self.fit_one_subject(subject, start_fit)
            errors.append(error)
        return np.mean(errors)

    def save_results(self, version: str, mean_error: float):
        """Save the fitting results to a text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fitting_results_PT_DD_DIFFEV_{version}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Fitting Results for Version: {version}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Individual Subject Errors:\n")
            for subject, error in sorted(self.subject_errors.items()):
                f.write(f"Subject {subject}: {error:.6f}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Mean Error Across All Subjects: {mean_error:.6f}\n")
            
            f.write("\nBest Fit Parameters:\n")
            for subject, params in sorted(self.best_fits.items()):
                f.write(f"\nSubject {subject}:\n")
                f.write(f"a: {params.a:.6f}\n")
                f.write(f"b: {params.b:.6f}\n")
                f.write(f"g: {params.g:.6f}\n")
                f.write(f"l: {params.l:.6f}\n")
                f.write(f"epsilon_gains: {params.epsilon_gains:.6f}\n")
                f.write(f"epsilon_losses: {params.epsilon_losses:.6f}\n")

def main(version: str):
    model = PT_TW_DDModel()
    data = load_dataset()
    model.data = data
    start_fit = Parameters(a=0.5, b=0.5, g=1.0, l=2.25, epsilon_gains=0.5, epsilon_losses=0.5)
    mean_error = model.fit_all_subjects(start_fit)
    model.save_results(version, mean_error)

if __name__ == '__main__':
    version = "tw_dd_v4"
    main(version=version)

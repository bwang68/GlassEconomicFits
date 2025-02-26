import numpy as np
import pandas as pd 
from functools import lru_cache
import math
from dataclasses import dataclass
from scipy.optimize import differential_evolution, direct, basinhopping
from typing import List, Tuple
import sys
from pathlib import Path
from parameter_visualization_pt import (
    visualize_participant_results,
    plot_parameter_changes
)

# Price probabilities dictionary
p = {
    1: 0.03,
    2: 0.06, 
    3: 0.09,
    4: 0.12, 
    5: 0.14,
    6: 0.11,
    7: 0.09,
    8: 0.08,
    9: 0.07,
    10: 0.06,
    11: 0.05,
    12: 0.04,
    13: 0.03,
    14: 0.02,
    15: 0.01
}

I = 15  # Maximum price

@dataclass(frozen=True)
class Parameters:
    a: float  # alpha
    b: float  # beta 
    l: float  # lambda
    g: float  # gamma

def prelec(p: float, gamma: float) -> float:
    """Probability weighting function"""
    return math.exp(-(-math.log(p)) ** gamma)

class PT_Model:
    def __init__(self):
        self.data = None
        self.cutoffs = None

    def PTv3(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Regular PT model without time window"""
        util = 0
        
        # Calculate gains
        for j in range(c_o[day-1], price):
            util += self.obj_gain(day, price, sold, params, c_o, j)

        # Calculate losses  
        for j in range(max(price+1,c_o[day-1]), I+1):
            util -= self.obj_loss(day, price, sold, params, c_o, j)

        # Add probability weighted terms
        for k in range(1, day+2):
            h_prob = self.h_probPTv2(day, k-1, c_o, params.g)
            
            gain_prob = 0
            for j in range(c_o[day-k], price):
                gain_prob += self.obj_gain(day, price, sold, params, c_o, j)

            loss_prob = 0  
            for j in range(max(price+1,c_o[day-k]), I+1):
                loss_prob += self.obj_loss(day, price, sold, params, c_o, j)

            util += h_prob * (gain_prob - loss_prob)

        return util

    def obj_gain(self, day: int, price: int, sold: int, params: Parameters, 
                c_o: List[int], j: int) -> float:
        """Calculate gain component"""
        gain = sold*(price-j)
        gain = gain**params.a
        gain *= prelec(p[j], params.g)
        return gain

    def obj_loss(self, day: int, price: int, sold: int, params: Parameters,
                c_o: List[int], j: int) -> float:
        """Calculate loss component"""
        loss = sold*(j-price) 
        loss = loss**params.b
        loss *= params.l * prelec(p[j], params.g)
        return loss

    def h_probPTv2(self, day: int, f: int, c_o: List[int], g: float) -> float:
        """Calculate h probability"""
        prob = 1.0
        for h in range(1,f+1):
            prob *= sum(prelec(p[j],g) for j in range(1,c_o[day-h]))
        return prob

    def cutoff_pt_list(self, params: Parameters) -> List[int]:
        """Calculate cutoff prices for each day"""
        cutoffs = [1]
        for day in range(1,68):
            if day == 0:
                continue
            price = cutoffs[day-1]
            
            prosp = self.PTv3(day, price, 1, params, cutoffs)
    

            while (prosp <= 0 and price < I):
                price += 1
                
                prosp = self.PTv3(day, price, 1, params, cutoffs)

            cutoffs.append(price)
        return cutoffs

    def max_units(self, day: int, price: int, stored: int, params: Parameters, cutoffs: List[int]) -> int:
        """Determine optimal number of units to sell"""
        sells = []
        pred = 0
        greatest = float('-inf') #uses negative infinity

        for units in range(1, stored+1):
            prosp = self.PTv3(day, price, units, params, cutoffs)

            if prosp > greatest:
                greatest = prosp
                pred = units
                sells.append(units)

        # add validation
        if greatest == float('-inf'):
            return 0 
        
        return pred

    def predict_sales(self, participant: int, params: Parameters) -> List[int]:
        """Make sales predictions for a participant"""
        predictions = []
        cutoffs = self.cutoff_pt_list(params)

        for day in range(68):
            vals = self.info_return(day, participant) # Get participant data
            stored = vals[1]
            price = vals[2]
            pred = 0

            if day == 0:
                pred = stored
            else:
                if price >= cutoffs[day]:
                    pred = self.max_units(day, price, stored, params, cutoffs)

            predictions.append(pred)

        return predictions

    def count_error(self, params_array: np.ndarray, participant: int) -> float:
        """Calculate prediction error"""
        # Convert numpy array to Parameters object
        params = Parameters(
            a=float(params_array[0]),
            b=float(params_array[1]), 
            l=float(params_array[2]),
            g=float(params_array[3])
        )
        
        preds = self.predict_sales(participant, params)
        total = 0
        
        for day in range(68):
            vals = self.info_return(day, participant)
            sold = vals[0]
            pred = preds[day]
            day_err = abs(sold-pred)
            total += day_err
            
        return total

    def fit_participant(self, participant: int):
        """Fit model parameters for a participant using basinhopping"""
        bounds = [(0, 1), (0, 1), (1, 2), (0, 1)]
        x0 = np.array([0.5, 0.5, 1.0, 0.5]) # Initial guess
        
        # Pass args through minimizer_kwargs
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "args": (participant,)  # Pass args here instead
        }

        result = basinhopping(
            func=self.count_error,
            x0=x0, 
            niter=500,
            T=1.0,              # Temperature parameter
            stepsize=0.25,      # Step size for random displacement
            minimizer_kwargs=minimizer_kwargs,
            interval=50,        # Interval for stepsize adjustment
            niter_success=20,   # Successful iterations needed
            disp=True,          # Display progress
            seed=42            
        )

        print(f"\nParticipant {participant} optimization:") 
        print(f"Success: {result.success}")
        print(f"Global minimum: {result.lowest_optimization_result.fun}")
        print(f"Number of iterations: {result.nit}")
        
        return result.x

    def info_return(self, day: int, participant: int) -> Tuple[int, int, int]:
        """Get participant data for a day"""
        if participant not in self.data:
            raise KeyError(f"Participant {participant} not found in data. Available participants: {sorted(self.data.keys())}")
            
        curr_df = self.data[participant]
        try:
            row = curr_df[curr_df['Day'] == day].iloc[0]
            return (row['Sold'], row['Stored'], row['Price'])
        except Exception as e:
            print(f"Error accessing data for participant {participant}, day {day}")
            print("DataFrame head:\n", curr_df.head())
            raise

    def test_parameter_set(self, participant: int, params: Parameters) -> float:
        """Test a single set of parameters for a participant"""
        preds = self.predict_sales(participant, params)
        total_error = 0
        
        for day in range(68):
            vals = self.info_return(day, participant)
            sold = vals[0]
            pred = preds[day]
            day_err = abs(sold-pred)
            total_error += day_err
            
        return total_error

def load_dataset() -> pd.DataFrame:
    """Load the participant data"""
    try:
        # Update path to your data file
        data = pd.read_csv("/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv")
        print("Original data shape:", data.shape)
        print("Available columns:", data.columns.tolist())
        print("First few rows:\n", data.head())
        
        # Convert to long format with Subject, Day, Sold, Stored, Price columns
        subjects = []
        days = []
        sold = []
        stored = []
        prices = []
        
        for idx, row in data.iterrows():
            subject = row['Subject']
            for day in range(1, 69):  # Days 1-68
                subjects.append(subject)
                days.append(day-1)  # Convert to 0-based index
                sold.append(row[f'Sold{day}'])
                stored.append(row[f'Stored{day}'])
                prices.append(row[f'Price{day}'])
        
        # Create new dataframe
        long_data = pd.DataFrame({
            'Subject': subjects,
            'Day': days,
            'Sold': sold,
            'Stored': stored,
            'Price': prices
        })
        
        # Create dictionary with subject-specific dataframes
        data_dict = {}
        for subject in long_data['Subject'].unique():
            subject_data = long_data[long_data['Subject'] == subject].sort_values('Day').reset_index(drop=True)
            data_dict[subject] = subject_data
            
        print(f"Successfully loaded dataset with {len(data_dict)} participants")
        print("Available participants:", sorted(data_dict.keys()))
        return data_dict
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def load_parameter_sets() -> pd.DataFrame:
    """Load parameter sets from Excel file"""
    try:
        params_df = pd.read_excel("/Users/brianwang/python-projs/GlassLab/data/FitValuesTest.xlsx")
        # Ensure Subject is numeric type
        params_df['Subject'] = pd.to_numeric(params_df['Subject'])
        print(f"Loaded {len(params_df)} parameter sets")
        print(f"Available subjects in parameter sets: {sorted(params_df['Subject'].unique())}")
        return params_df
    except Exception as e:
        print(f"Error loading parameter sets: {e}")
        sys.exit(1)

def test_all_participants(model: PT_Model, parameter_sets: pd.DataFrame) -> dict:
    """Test parameter sets for participants, but only using rows matching their subject number"""
    results = {}
    
    # Get all unique subjects from both datasets
    data_subjects = set(model.data.keys())
    param_subjects = set(parameter_sets['Subject'].unique())
    
    print("\nData subjects:", sorted(data_subjects))
    print("Parameter subjects:", sorted(param_subjects))
    print("Subjects in both sets:", sorted(data_subjects & param_subjects))
    
    # Use intersection of both sets
    participants = sorted(data_subjects & param_subjects)
    
    for participant in participants:
        participant_num = int(participant)
        participant_params = parameter_sets[parameter_sets['Subject'] == participant_num]
        
        if participant_params.empty:
            print(f"\nWarning: No parameter sets found for participant {participant_num}")
            continue
            
        results[participant_num] = []
        print(f"\nTesting participant {participant_num} with {len(participant_params)} parameter sets")
        
        for idx, row in participant_params.iterrows():
            params = Parameters(
                a=float(row['alphaNoTW']),
                b=float(row['betaNoTW']),
                g=float(row['gammaNoTW']),
                l=float(row['lambdaNoTW'])
            )
            
            error = model.test_parameter_set(participant, params)
            results[participant].append({
                'set_index': idx,
                'params': params,
                'error': error
            })
            print(f"Parameter set {idx}: error = {error:.2f}")
            
    return results

def parse_parameter_file(filepath: str) -> dict:
    """Parse Best PT Fits text file and return parameters by participant"""
    params_dict = {}
    current_participant = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().lower()  # Convert to lowercase
            if line.startswith('participant'):
                current_participant = int(line.split()[1].rstrip(':'))
                params_dict[current_participant] = {}
            elif any(line.startswith(param) for param in ('alpha:', 'beta:', 'lambda:', 'gamma:')):
                param_name, value = line.split(':')
                param_name = param_name.strip()
                value = value.strip()
                params_dict[current_participant][param_name] = float(value)
    
    return params_dict

def generate_predictions_df(input_csv: str, params_file: str, output_csv: str):
    """Generate predictions and create new CSV with prediction columns"""
    # Load original data
    original_df = pd.read_csv(input_csv)
    
    # Load parameters
    params_dict = parse_parameter_file(params_file)
    
    # Initialize model
    model = PT_Model()
    model.data = load_dataset()
    
    # Create prediction columns and initialize to 0
    for day in range(1, 69):
        original_df[f'Predicted{day}'] = 0
    
    # Generate predictions for each subject
    for subject in original_df['Subject'].unique():
        if subject in params_dict:
            p = params_dict[subject]
            params = Parameters(
                a=p['alpha'],
                b=p['beta'],
                l=p['lambda'],
                g=p['gamma']
            )
            
            predictions = model.predict_sales(subject, params)
            
            for day in range(1, 69):
                original_df.loc[original_df['Subject'] == subject, f'Predicted{day}'] = predictions[day-1]
    
    # Reorder columns to interleave Predicted columns with existing day columns
    new_columns = ['Subject']
    for day in range(68, 0, -1):  # Go from day 68 to 1
        new_columns.extend([
            f'Stored{day}',
            f'Price{day}',
            f'Sold{day}',
            f'Predicted{day}'
        ])
    
    # Reorder the columns and save
    original_df = original_df[new_columns]
    original_df.to_csv(output_csv, index=False)

def main():
    # Initialize model
    model = PT_Model()
    
    # Load data
    data = load_dataset()
    model.data = data
    
    # Load parameter sets
    parameter_sets = load_parameter_sets()
    
    # Verify Subject column exists
    if 'Subject' not in parameter_sets.columns:
        print("Error: 'Subject' column not found in parameter sets file")
        sys.exit(1)
    
    # Test matching parameter sets
    results = test_all_participants(model, parameter_sets)
    
    # Print summary
    print("\nTesting Results Summary:")
    print("-" * 50)
    
    for participant, participant_results in results.items():
        if not participant_results:
            print(f"\nParticipant {participant}: No parameter sets tested")
            continue
            
        print(f"\nParticipant {participant}:")
        best_result = min(participant_results, key=lambda x: x['error'])
        print(f"Best parameter set (index {best_result['set_index']}):")
        print(f"Alpha: {best_result['params'].a:.4f}")
        print(f"Beta: {best_result['params'].b:.4f}")
        print(f"Lambda: {best_result['params'].l:.4f}")
        print(f"Gamma: {best_result['params'].g:.4f}")
        print(f"Error: {best_result['error']:.2f}")

    # Update paths as needed
    input_csv = "/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv"
    params_file = "/Users/brianwang/python-projs/GlassLab/data/Best PT Fits.txt"
    output_csv = "/Users/brianwang/python-projs/GlassLab/PT_TW_DD_WITH_PREDICTIONS.csv"
    
    generate_predictions_df(input_csv, params_file, output_csv)

if __name__ == "__main__":
    main()



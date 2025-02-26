import pandas as pd
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import math
from dataclasses import dataclass
from scipy.optimize import least_squares, basinhopping
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Tuple

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

I = 15 # maximum price

@dataclass(frozen=True)
class Parameters:
    a: float = 1.0
    b: float = 1.0
    g: float = 1.0
    l: float = 1.0
    epsilon: float = 1.0
    tw: int = 68

def prelec(p: float, gamma: float) -> float:
    """probability weighting function with safety checks"""
    if p <= 0:
        return 0.0
    try:
        return math.exp(-((-math.log(p)) ** gamma))
    except (ValueError, OverflowError):
        return 0.0

class PT_DD_Model:
    def __init__(self):
        self.data = None
        self.cutoffs = None
    
    def PT_TW_DD(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Prospect Theory with delay discounting model"""
        try:
            t = int(params.tw)

            if t > day:  # if time window is greater than the day
                return self.PTv3(day, price, sold, params, c_o)
            
            # Calculate epsilon denominator once
            epsilon_sum = sum(params.epsilon * (k - 1) for k in range(1, c_o[t-1]))
            epsilon_factor = max(1.0, 1 + epsilon_sum)  # Ensure positive denominator
            
            # Calculate probabilities safely
            prob_sum = max(0.0, sum(p.get(k, 0.0) for k in range(1, c_o[t-1])))
            
            gain = 0.0
            loss = 0.0
            
            # Gains calculation
            for j in range(c_o[t-1], price):
                if j in p:
                    gain_value = sold * (price-j) / epsilon_factor
                    if gain_value > 0:
                        prob = p[j] * (1 + prob_sum)
                        gain += prelec(prob, params.g) * (gain_value ** params.a)
            
            # Losses calculation
            for j in range(max(price+1, c_o[t-1]), I+1):
                if j in p:
                    loss_value = sold * (j-price) / epsilon_factor
                    if loss_value > 0:
                        prob = p[j] * (1 + prob_sum)
                        loss += prelec(prob, params.g) * params.l * (loss_value ** params.b)
            
            return gain - loss
        except Exception as e:
            print(f"Error in PT_TW_DD: {e}")
            return 0.0
        
    def h_probPTv2(self, day: int, f: int, c_o: List[int], g: float) -> float:
        """calculate h probability for PT"""
        prob = 1.0
        for h in range(1, f + 1):
            prob *= sum(prelec(p[j],g) for j in range(1, c_o[day-h]))

        return prob

    def PTv3(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Prospect Theory model"""
        util = 0 
        t = int(params.tw)
        
        # Calculate epsilon denominator once
        epsilon_sum = sum(params.epsilon * (k - 1) for k in range(1, c_o[t-1]))
        epsilon_factor = 1 + epsilon_sum if epsilon_sum > 0 else 1

        # gains 
        for j in range(c_o[day-1], price):
            util += ((sold * (price-j) / epsilon_factor) ** params.a) * prelec(p[j], params.g)

        # losses
        for j in range(max(price+1, c_o[day-1]), I+1):
            util -= ((sold * (j-price) / epsilon_factor) ** params.b) * params.l * prelec(p[j], params.g)

        # add probability weight 
        for k in range(1, day+2):
            h_prob = self.h_probPTv2(day, k-1, c_o, params.g)

            gain_prob = 0 
            for j in range(c_o[day-k], price):
                gain_prob += ((sold * (price-j) / epsilon_factor) ** params.a) * prelec(p[j], params.g)

            loss_prob = 0
            for j in range(max(price+1, c_o[day-k]), I+1):
                loss_prob += ((sold * (j-price) / epsilon_factor) ** params.b) * params.l * prelec(p[j], params.g)
    
            util += h_prob * (gain_prob - loss_prob)

        return util
    

    def cutoff_pt_tw_list(self, params: Parameters) -> List[int]:
        """calculate cutoffs for PT_TW_DD model"""
        # Initialize with 68 days of cutoffs
        c_o = [1] * 68  # Initialize all days with 1
        
        for day in range(1, 68):  # Start from day 1
            price = c_o[day-1]
            prosp = 0
            
            if day < params.tw:
                prosp = self.PTv3(day, price, 1, params, c_o)
            else: 
                prosp = self.PT_TW_DD(day, price, 1, params, c_o)

            while price < I and prosp <= 0:
                price += 1
                if day < params.tw:
                    prosp = self.PTv3(day, price, 1, params, c_o)
                else:
                    prosp = self.PT_TW_DD(day, price, 1, params, c_o)
            
            c_o[day] = price

        return c_o

    def max_units(self, day: int, price: int, stored: int, params: Parameters, cutoffs: List[int]) -> int:
        """calculate maximum units that can be sold"""
        units_sold = []
        pred = 0
        greatest = float('-inf') 

        for units in range(1, stored+1):
            prosp = self.PT_TW_DD(day, price, units, params, cutoffs)

            if prosp > greatest:
                greatest = prosp
                pred = units
                units_sold.append(pred)

        return pred
    
    ### NOT USED IN PT_DD MODEL, TW fixed at 68
    # def find_optimal_tw(self, participant: int, other_params: List[float]) -> Tuple[float, float]:
        """Find optimal time window for a participant using ternary search
        Returns optimal time window and corresponding error"""

        def get_error_for_tw(tw: int) -> float:
            return self.count_error(other_params, participant, tw=float(tw))
       
        left, right = 2, 68
        best_tw = None
        best_error = float('inf')
        min_improvement = 1.0 # error has to improve by this much AT LEAST

        prev_bounds = None
        repeat_count = 0

        while right - left > 1:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3

            error1 = get_error_for_tw(mid1)
            error2 = get_error_for_tw(mid2)

            current_bounds = (left, right)
            if current_bounds == prev_bounds:
                repeat_count += 1
                if repeat_count >= 3:
                    print(f"Same plataeu, using best values")
                    return best_tw, best_error
                else:
                    prev_bounds = current_bounds
                    repeat_count = 0

            # update best found so far
            if error1 < best_error:
                best_error = error1
                best_tw = mid1
            if error2 < best_error:
                best_error = error2
                best_tw = mid2

            print(f"Testing tw={mid1}: error={error1:.2f}, tw={mid2}: error={error2:.2f}")

            if error1 < error2:
                right = mid2 - 1 # force bounds to chagne
            else:
                left = mid1 + 1

            if right - left <= 2:
                break
        
        return best_tw, best_error
    
### now all the calculations set up, we need to load the data and get the info
    def info_return(self, day: int, participant: int) -> Tuple[int, int, int]:
        """Get participant data for a day"""
        try:
            if participant not in self.data:
                raise KeyError(f"Participant {participant} not found in data")
            
            curr_df = self.data[participant]
            row = curr_df.loc[curr_df['Day'] == day]
            
            if row.empty:
                raise KeyError(f"Day {day} not found for participant {participant}")
                
            return (
                int(row['Sold'].iloc[0]),
                int(row['Stored'].iloc[0]),
                int(row['Price'].iloc[0])
            )
        except Exception as e:
            print(f"Error in info_return: Participant={participant}, Day={day}")
            print(f"DataFrame shape: {curr_df.shape if 'curr_df' in locals() else 'N/A'}")
            print(f"Available days: {curr_df['Day'].unique().tolist() if 'curr_df' in locals() else 'N/A'}")
            raise e

    def predict_sales(self, participant: int, params: Parameters) -> List[int]:
        """Predict sales for a participant"""
        predictions = []
        cutoffs = self.cutoff_pt_tw_list(params)

        for day in range(68):
            vals = self.info_return(day, participant)
            stored = vals[1]
            price = vals[2]
            pred = 0

            if day == 0:
                pred = stored
            else:
                if price >= cutoffs[day]:  # Now this index will always be valid
                    pred = self.max_units(day, price, stored, params, cutoffs)
            
            predictions.append(pred)

        return predictions
    
    def count_error(self, params_arry: np.ndarray, participant: int, tw: float = None) -> float:
        """Count error for a participant with additional checks"""
        try:
            # Ensure parameters are within valid ranges
            if any(np.isnan(params_arry)) or any(np.isinf(params_arry)):
                return float('inf')
                
            if tw is None:
                params = Parameters(
                    a = max(0.01, min(0.99, float(params_arry[0]))),
                    b = max(0.01, min(0.99, float(params_arry[1]))),
                    l = max(1.0, min(2.0, float(params_arry[2]))),
                    g = max(0.01, min(0.99, float(params_arry[3]))),
                    tw = 14.0
                )
            else:
                params = Parameters(
                    a = max(0.01, min(0.99, float(params_arry[0]))),
                    b = max(0.01, min(0.99, float(params_arry[1]))),
                    l = max(1.0, min(2.0, float(params_arry[2]))),
                    g = max(0.01, min(0.99, float(params_arry[3]))),
                    tw = float(tw)
                )

            preds = self.predict_sales(participant, params)
            total = sum(abs(self.info_return(day, participant)[0] - pred) 
                    for day, pred in enumerate(preds))
            
            return float(total)
        except Exception as e:
            print(f"Error in count_error: {e}")
            return float('inf')
    
    def fit_participant(self, participant: int):
        """Fit model parameters for a participant using basinhopping"""
        bounds = [(0, 1), (0, 1), (1, 2), (0, 1), (68, 68)]
        x0 = np.array([0.5, 0.5, 1.0, 0.5, 68]) # Initial guess
        
        def callback(x, f, accept):
            print(f"Params: {x}, Error: {f}")
            return False

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
            T=2.0,              # Temperature parameter
            stepsize=0.5,      # Step size for random displacement
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,
            interval=50,        # Interval for stepsize adjustment
            niter_success=20,   # Successful iterations needed
            disp=True,          # Display progress
            seed=420    
        )

        #find optimal tw using other parameters NOT USED IN PT_DD MODEL
        # best_tw, best_error = self.find_optimal_tw(participant, result.x)


        print(f"\nParticipant {participant} optimization:") 
        print(f"Success: {result.success}")
        print(f"Global minimum: {result.lowest_optimization_result.fun}")
        print(f"Number of iterations: {result.nit}")
        
        return np.append(result.x)
    
# class finished, now we load data and run model

def load_dataset() -> pd.DataFrame:
    """Load the participant data"""
    try:
        data = pd.read_csv("/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv")
        print("Original data shape:", data.shape)
        print("Available columns:", data.columns.tolist())
        
        # Convert to long format with Subject, Day, Sold, Stored, Price columns
        long_data = []
        
        for _, row in data.iterrows():
            subject = row['Subject']
            for day in range(1, 69):  # Days 1-68
                long_data.append({
                    'Subject': subject,
                    'Day': day-1,  # Convert to 0-based index
                    'Sold': int(row[f'Sold{day}']),
                    'Stored': int(row[f'Stored{day}']),
                    'Price': int(row[f'Price{day}'])
                })
        
        # Create dataframe
        long_data = pd.DataFrame(long_data)
        
        # Create dictionary with subject-specific dataframes
        data_dict = {
            subject: group.copy() for subject, group in long_data.groupby('Subject')
        }
        
        # Validate data
        for subject, df in data_dict.items():
            print(f"Participant {subject} has {len(df)} days of data")
            if len(df) != 68:
                print(f"Warning: Participant {subject} does not have 68 days of data!")
                
        return data_dict
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def fit_all_participants(model: PT_DD_Model) -> dict:
    """Fit model parameters for all participants"""
    results = {}
    
    participants = sorted(model.data.keys())
    total_participants = len(participants)
    
    print(f"\nParticipant list: {participants}")
    print(f"\nTotal participants to fit: {total_participants}")
    print("\nStarting optimization...\n")
    
    for i, participant in enumerate(participants, 1):
        print(f"\nFitting participant {participant} ({i}/{total_participants})")
        try:
            # Fit participant
            best_fit = model.fit_participant(participant)
            
            # Convert best_fit numpy array to regular Python types
            params = {
                'alpha': float(best_fit[0]),
                'beta': float(best_fit[1]),
                'lambda': float(best_fit[2]),
                'gamma': float(best_fit[3]),
                'time_window': int(round(best_fit[4])),
                'error': float(model.count_error(best_fit, participant))
            }
            
            results[participant] = params
            print(f"Completed participant {participant} with error: {params['error']}")
            
        except Exception as e:
            print(f"Error fitting participant {participant}: {str(e)}")
            results[participant] = {'error': str(e)}
    
    return results

def main():
    # Initialize model
    model = PT_DD_Model()
    
    # Load data
    data = load_dataset()
    model.data = data
    
    # Fit all participants
    results = fit_all_participants(model)
    
    # Print summary
    print("\nFitting Results Summary:")
    print("-" * 50)
    
    successful_fits = 0
    total_error = 0
    
    for participant, result in results.items():
        if 'error' in result and isinstance(result['error'], (int, float)):
            try:
                print(f"\nParticipant {participant}:")
                print(f"Alpha: {result['alpha']:.4f}")
                print(f"Beta: {result['beta']:.4f}")
                print(f"Lambda: {result['lambda']:.4f}")
                print(f"Gamma: {result['gamma']:.4f}")
                print(f"Time Window: {result['time_window']}")
                print(f"Error: {result['error']:.2f}")
                successful_fits += 1
                total_error += result['error']
            except Exception as e:
                print(f"\nParticipant {participant}: Error processing results - {e}")
        else:
            print(f"\nParticipant {participant}: Failed to fit - {result.get('error', 'Unknown error')}")
    
    if successful_fits > 0:
        avg_error = total_error / successful_fits
        print(f"\nSuccessfully fit {successful_fits} participants")
        print(f"Average error: {avg_error:.2f}")
    else:
        print("\nNo successful fits")

if __name__ == "__main__":
    main()
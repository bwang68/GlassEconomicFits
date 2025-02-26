import numpy as np
import pandas as pd 
from functools import lru_cache
import math
from dataclasses import dataclass
from scipy.optimize import differential_evolution, direct, basinhopping
from typing import List, Tuple
import sys
from pathlib import Path

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
    tw: float # time window

def prelec(p: float, gamma: float) -> float:
    """Probability weighting function"""
    return math.exp(-(-math.log(p)) ** gamma)

class PT_TW_Model:
    def __init__(self):
        self.data = None
        self.cutoffs = None

    def PT_TW(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Prospect Theory with Time Window utility calculation"""
        t = int(params.tw)

        if t > day:  # If time window larger than current day
            # Use regular PT model
            return self.PTv3(day, price, sold, params, c_o)
        else:
            # Calculate gains with time window
            gain = sum(prelec(p[j] + sum((sum(p[k] for k in range(1,c_o[t-1])))**h 
                     for h in range(1,t-1)) * p[j], params.g) * 
                     (sold*(price-j))**params.a 
                     for j in range(c_o[t-1],price))
            
            # Calculate losses with time window
            loss = sum(prelec(p[j] + sum((sum(p[k] for k in range(1,c_o[t-1])))**h 
                     for h in range(1,t-1)) * p[j], params.g) *
                     params.l * (sold*(j-price))**params.b 
                     for j in range(max(c_o[t-1],price+1),I+1))

            return gain - loss

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

    def cutoff_pt_tw_list(self, params: Parameters) -> List[int]:
        """Calculate cutoff prices for each day"""
        cutoffs = [1]
        for day in range(1,68):
            if day == 0:
                continue
            price = cutoffs[day-1]
            
            if day < params.tw:
                prosp = self.PTv3(day, price, 1, params, cutoffs)
            else:
                prosp = self.PT_TW(day, price, 1, params, cutoffs)

            while (prosp <= 0 and price < I):
                price += 1
                if day < params.tw:
                    prosp = self.PTv3(day, price, 1, params, cutoffs)
                else:
                    prosp = self.PT_TW(day, price, 1, params, cutoffs)

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
    

    # figure out optimal tw

    def find_optimal_tw(self, participant: int, other_params: List[float]) -> Tuple[float, float]:
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



    def predict_sales(self, participant: int, params: Parameters) -> List[int]:
        """Make sales predictions for a participant"""
        predictions = []
        cutoffs = self.cutoff_pt_tw_list(params)

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

    def count_error(self, params_array: np.ndarray, participant: int, tw: float = None) -> float:
        """Calculate prediction error"""
        
        if tw is None:
            params = Parameters(
                a=float(params_array[0]),
                b=float(params_array[1]), 
                l=float(params_array[2]),
                g=float(params_array[3]),
                tw=14.0
            )
        else:
            # when finding the best tw

            params = Parameters(
                a=float(params_array[0]),
                b=float(params_array[1]), 
                l=float(params_array[2]),
                g=float(params_array[3]),
                tw=float(tw) 
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
        x0 = np.array([1.0, 1.0, 1.755, 1.0]) # Initial guess
        
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

        #find optimal tw using other parameters
        best_tw, best_error = self.find_optimal_tw(participant, result.x)


        print(f"\nParticipant {participant} optimization:") 
        print(f"Success: {result.success}")
        print(f"Global minimum: {result.lowest_optimization_result.fun}")
        print(f"Number of iterations: {result.nit}")
        
        return np.append(result.x, best_tw)

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

def fit_all_participants(model: PT_TW_Model) -> dict:
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
    model = PT_TW_Model()
    
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



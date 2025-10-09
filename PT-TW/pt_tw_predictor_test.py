import numpy as np
import pandas as pd 
from functools import lru_cache
import math
from dataclasses import dataclass
from scipy.optimize import differential_evolution, direct, basinhopping
from typing import List, Tuple, Optional
import sys
from pathlib import Path
import os

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
    tw: float = 67  # time window locked at 67


# Parameter bounds for optimization: (min, max)
PARAM_BOUNDS: List[Tuple[float, float]] = [
    (0.01, 2.0),   # alpha
    (0.01, 2.0),   # beta
    (0.05, 5.0),   # lambda
    (0.01, 2.0),   # gamma
    (1.0, 68.0),   # time window
]

def prelec(p: float, gamma: float) -> float:
    """Probability weighting function"""
    return math.exp(-(-math.log(p)) ** gamma)

class PT_TW_Model:
    def __init__(self):
        self.data = None
        self.cutoffs = None
        # Mean proportional error configuration (defaults preserve current behavior)
        # - denominator: 'stored' or 'sold'
        # - skip_if_stored_zero: skip days with stored == 0
        # - day range uses 0-based indices as loaded in load_dataset(): [0, 67]
        self.mpe_denominator: str = os.getenv("PTTW_MPE_DENOM", "stored").lower()
        self.mpe_skip_if_stored_zero: bool = os.getenv("PTTW_MPE_SKIP_STORED_ZERO", "1") not in ("0", "false", "False")
        # Start day (0 includes initial day; Excel often starts at 1)
        try:
            self.mpe_day_start: int = int(os.getenv("PTTW_MPE_DAY_START", "0"))
        except ValueError:
            self.mpe_day_start = 0
        try:
            self.mpe_day_end: int = int(os.getenv("PTTW_MPE_DAY_END", "67"))
        except ValueError:
            self.mpe_day_end = 67

    @staticmethod
    def _safe_cutoff(c_o: List[int], idx: int) -> int:
        """Return cutoff at non-negative index; clamp negatives to 0 to avoid Python's negative indexing."""
        return c_o[0] if idx < 0 else c_o[idx]

    def PT_TW(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Prospect Theory with Time Window utility calculation"""
        t = int(params.tw)

        if t > day:  # If time window larger than current day
            # Use regular PT model
            return self.PTv3(day, price, sold, params, c_o)
        else:

            lower_cutoff = self._safe_cutoff(c_o, day - 1)
            # Calculate gains with time window
            gain = 0.0
            for j in range(lower_cutoff, price):
                prob_horizon = sum(
                    (sum(p[k] for k in range(1, lower_cutoff))) ** h
                    for h in range(1, t - 1)
                )
                weighted_prob = prelec(p[j] + prob_horizon * p[j], params.g)
                gain += weighted_prob * (sold * (price - j)) ** params.a
            
            # Calculate losses with time window
            loss = 0.0
            for j in range(max(lower_cutoff, price+1), I+1):
                prob_horizon = sum(
                    (sum(p[k] for k in range(1, lower_cutoff))) ** h
                    for h in range(1, t - 1)
                )
                weighted_prob = prelec(p[j] + prob_horizon * p[j], params.g)
                loss += weighted_prob * params.l * (sold * (j - price)) ** params.b

            return gain - loss

    def PTv3(self, day: int, price: int, sold: int, params: Parameters, c_o: List[int]) -> float:
        """Regular PT model without time window"""
        util = 0
        
        # Calculate gains
        for j in range(self._safe_cutoff(c_o, day-1), price):
            util += self.obj_gain(day, price, sold, params, c_o, j)

        # Calculate losses  
        for j in range(max(price+1, self._safe_cutoff(c_o, day-1)), I+1):
            util -= self.obj_loss(day, price, sold, params, c_o, j)

        # Add probability weighted terms
        for k in range(1, day+2):
            h_prob = self.h_probPTv2(day, k-1, c_o, params.g)
            
            gain_prob = 0
            for j in range(self._safe_cutoff(c_o, day-k), price):
                gain_prob += self.obj_gain(day, price, sold, params, c_o, j)

            loss_prob = 0  
            for j in range(max(price+1, self._safe_cutoff(c_o, day-k)), I+1):
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
            prob *= sum(prelec(p[j], g) for j in range(1, self._safe_cutoff(c_o, day-h)))
        return prob

    def cutoff_pt_tw_list(self, params: Parameters) -> List[int]:
        """Calculate cutoff prices for each day"""
        cutoffs = [1]
        for day in range(1,68):
            if day == 0:
                continue
            price = cutoffs[day-1]
            
            prosp = self.PT_TW(day, price, 1, params, cutoffs)

            while (prosp <= 0 and price < I):
                price += 1
                prosp = self.PT_TW(day, price, 1, params, cutoffs)

            cutoffs.append(price)
        return cutoffs

    def max_units(self, day: int, price: int, stored: int, params: Parameters, cutoffs: List[int]) -> int:
        """Determine optimal number of units to sell"""
        sells = []
        pred = 0
        greatest = float('-inf') #uses negative infinity

        for units in range(0, max(0, stored) + 1):
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

    def test_parameter_set(self, participant: int, params: Parameters) -> float:
        """Test a single set of parameters for a participant"""
        preds = self.predict_sales(participant, params)
        return self.mean_proportional_error(participant, preds)

    def _clamp_parameters(self, raw: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Project raw parameter vector into provided bounds."""
        sel_bounds = bounds or PARAM_BOUNDS
        clamped = np.empty_like(raw, dtype=float)
        for idx, (low, high) in enumerate(sel_bounds):
            clamped[idx] = float(np.clip(raw[idx], low, high))
        return clamped

    def _vector_to_params(self, vec: np.ndarray) -> Parameters:
        """Convert vector to Parameters dataclass (after clamping)."""
        vec = self._clamp_parameters(vec)
        return Parameters(
            a=float(vec[0]),
            b=float(vec[1]),
            l=float(vec[2]),
            g=float(vec[3]),
            tw=float(vec[4]),
        )

    def _build_init_population(
        self,
        base_vec: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Construct a warm-start population for differential evolution around base vector."""
        sel_bounds = bounds or PARAM_BOUNDS
        base = self._clamp_parameters(base_vec, sel_bounds)
        population = [base]

        # Add directional nudges (±5% of span or minimum epsilon)
        for idx, (low, high) in enumerate(sel_bounds):
            span = max(high - low, 1e-3)
            step = max(0.05 * span, 1e-3)

            plus = base.copy()
            plus[idx] += step
            population.append(self._clamp_parameters(plus, sel_bounds))

            minus = base.copy()
            minus[idx] -= step
            population.append(self._clamp_parameters(minus, sel_bounds))

        # Ensure diversity using reproducible random samples
        rng = np.random.default_rng(seed)
        required = max(len(sel_bounds) * 4, 12)
        while len(population) < required:
            sample = np.array([rng.uniform(low, high) for (low, high) in sel_bounds], dtype=float)
            population.append(sample)

        # Remove duplicates while preserving order
        unique = []
        seen = set()
        for vec in population:
            key = tuple(np.round(vec, 6))
            if key not in seen:
                seen.add(key)
                unique.append(vec)

        return np.vstack(unique)

    def optimize_parameters(
        self,
        participant: int,
        initial_params: Parameters,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_global_iter: int = 25,
        basin_steps: int = 5,
    ) -> Tuple[Parameters, float]:
        """Warm-started two-stage optimization (global + local) to minimize mean proportional error."""

        sel_bounds = bounds or PARAM_BOUNDS
        base_vec = np.array([
            initial_params.a,
            initial_params.b,
            initial_params.l,
            initial_params.g,
            initial_params.tw,
        ], dtype=float)

        def objective(vec: np.ndarray) -> float:
            params = self._vector_to_params(vec)
            return self.test_parameter_set(participant, params)

        init_population = self._build_init_population(base_vec, sel_bounds, seed=int(participant))

        try:
            de_result = differential_evolution(
                objective,
                sel_bounds,
                init=init_population,
                maxiter=max_global_iter,
                polish=False,
            )
            best_vec = self._clamp_parameters(de_result.x, sel_bounds)
            best_error = float(de_result.fun)
            best_params = self._vector_to_params(best_vec)
        except Exception as exc:
            print(f"Differential evolution failed for participant {participant}: {exc}")
            best_params = initial_params
            best_error = self.test_parameter_set(participant, best_params)
            best_vec = base_vec

        # Local refinement using basin hopping with L-BFGS-B minimizer
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": sel_bounds,
        }

        try:
            bh_result = basinhopping(
                objective,
                best_vec,
                minimizer_kwargs=minimizer_kwargs,
                niter=basin_steps,
                stepsize=0.25,
                seed=int(participant),
                disp=False,
            )
            bh_error = float(bh_result.fun)
            if bh_error < best_error:
                best_error = bh_error
                best_params = self._vector_to_params(bh_result.x)
        except Exception as exc:
            print(f"Basinhopping refinement failed for participant {participant}: {exc}")

        return best_params, best_error

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

    def mean_proportional_error(self, participant: int, predictions: List[int]) -> float:
        """Calculate mean proportional error with configurable denominator (default 'stored').

        Notes:
        - Denominator: self.mpe_denominator in {'stored','sold'} (default 'stored')
        - Skips days with stored == 0 if configured (self.mpe_skip_if_stored_zero)
        - Uses day range [self.mpe_day_start, self.mpe_day_end] inclusive (0-based)
        """
        total_error = 0.0
        valid_days = 0

        day_start = max(0, self.mpe_day_start)
        day_end = min(67, self.mpe_day_end)

        for day in range(day_start, day_end + 1):
            sold, stored, _ = self.info_return(day, participant)

            if self.mpe_skip_if_stored_zero and stored == 0:
                continue

            # Choose denominator per configuration
            if self.mpe_denominator == "sold":
                denom_val = sold
            else:
                denom_val = stored
            denom = max(1, int(denom_val) if denom_val is not None else 1)

            pred = predictions[day]
            day_err = abs(int(pred) - int(sold)) / denom
            total_error += day_err
            valid_days += 1

        if valid_days == 0:
            return float("inf")

        return total_error / valid_days

    def debug_participant_days(
        self,
        participant: int,
        params: Parameters,
        output_path: Optional[Path] = None,
        denominator: Optional[str] = None,
        skip_if_stored_zero: Optional[bool] = None,
        day_start: Optional[int] = None,
        day_end: Optional[int] = None,
        include_utilities: bool = False,
    ) -> Path:
        """Dump per-day debug info for a participant: sold, stored, price, pred, per-day error.

        Does not change model policy. Useful to reconcile with Excel per-day terms.
        """
        # Use current config unless overrides provided
        denom_sel = (denominator or self.mpe_denominator).lower()
        skip_zero = self.mpe_skip_if_stored_zero if skip_if_stored_zero is None else bool(skip_if_stored_zero)
        d_start = self.mpe_day_start if day_start is None else int(day_start)
        d_end = self.mpe_day_end if day_end is None else int(day_end)

        # Ensure path
        if output_path is None:
            output_path = Path(__file__).resolve().parents[1] / "output_files" / f"debug_participant_{participant}.csv"

        # Predictions (uses current max_units policy)
        preds = self.predict_sales(participant, params)

        lines = [
            [
                "Day",
                "Stored",
                "Price",
                "Sold",
                "Pred",
                "Denominator",
                "DayError",
                "CumError",
            ]
        ]

        total = 0.0
        n = 0

        for day in range(max(0, d_start), min(67, d_end) + 1):
            sold, stored, price = self.info_return(day, participant)
            if skip_zero and stored == 0:
                continue
            # Choose denominator per configuration
            denom_val = sold if denom_sel == "sold" else stored
            denom = max(1, int(denom_val) if denom_val is not None else 1)
            pred = int(preds[day])
            day_err = abs(pred - int(sold)) / denom
            total += day_err
            n += 1
            lines.append([day, int(stored), int(price), int(sold), pred, denom, f"{day_err:.6f}", f"{(total/n):.6f}"])

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(",".join(map(str, lines[0])) + "\n")
            for row in lines[1:]:
                f.write(",".join(map(str, row)) + "\n")

        return output_path

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
        data = pd.read_csv("/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv")
        print("\nRaw data participants:", sorted(data['Subject'].unique()))  # Debug print
        
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
        
        # Ensure Subject is numeric type
        long_data['Subject'] = pd.to_numeric(long_data['Subject'])
        
        # Before creating data_dict, print all unique subjects
        all_subjects = sorted(long_data['Subject'].unique())
        print("\nProcessed unique subjects:", all_subjects)
        print("Number of unique subjects:", len(all_subjects))
        
        # Create dictionary with subject-specific dataframes
        data_dict = {}
        for subject in all_subjects:  # Use all_subjects instead of unique() again
            subject_data = long_data[long_data['Subject'] == subject].sort_values('Day').reset_index(drop=True)
            data_dict[int(subject)] = subject_data  # Force int key
            
        print(f"\nCreated dictionary entries for subjects:", sorted(data_dict.keys()))
        return data_dict
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise  # Show full error trace

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

def test_all_participants(model: PT_TW_Model, parameter_sets: pd.DataFrame) -> dict:
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
        participant_num = int(participant)  # Ensure numeric
        # Filter parameter sets to only include rows matching this participant
        participant_params = parameter_sets[parameter_sets['Subject'] == participant_num]
        
        if participant_params.empty:
            print(f"\nWarning: No parameter sets found for participant {participant_num}")
            continue
            
        results[participant_num] = []
        print(f"\nTesting participant {participant_num} with {len(participant_params)} parameter sets")
        
        for idx, row in participant_params.iterrows():
            params = Parameters(
                a=float(row['alphaNew']),
                b=float(row['betaNew']),
                g=float(row['gammaNew']),
                l=float(row['lambdaNew']),
                tw=float(row['twNew'])
            )
            
            error = model.test_parameter_set(participant, params)
            results[participant].append({
                'set_index': idx,
                'params': params,
                'error': error
            })
            print(f"Parameter set {idx}: mean proportional error = {error:.4f}")

        # Warm-started optimization based on best initial set
        if results[participant]:
            best_seed = min(results[participant], key=lambda x: x['error'])
            print(
                f"\nOptimizing participant {participant} starting from set {best_seed['set_index']} "
                f"(error {best_seed['error']:.4f})"
            )
            try:
                optimized_params, optimized_error = model.optimize_parameters(participant, best_seed['params'])
                results[participant].append({
                    'set_index': 'optimized',
                    'params': optimized_params,
                    'error': optimized_error,
                    'source': 'warm-start-optimized'
                })
                print(
                    f"Optimized parameters: alpha={optimized_params.a:.4f}, beta={optimized_params.b:.4f}, "
                    f"lambda={optimized_params.l:.4f}, gamma={optimized_params.g:.4f}, tw={optimized_params.tw:.2f}"
                )
                print(f"Optimized mean proportional error = {optimized_error:.4f}")
            except Exception as exc:
                print(f"Optimization failed for participant {participant}: {exc}")
            
    return results

def parse_parameter_file(filepath: str) -> dict:
    """Parse Best PTTW Fits text file and return parameters by participant"""
    params_dict = {}
    current_participant = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().lower()  # Convert to lowercase
            if line.startswith('participant'):
                current_participant = int(line.split()[1].rstrip(':'))
                params_dict[current_participant] = {}
            elif any(line.startswith(param) for param in ('alpha:', 'beta:', 'lambda:', 'gamma:', 'time window:')):
                param_name, value = line.split(':')
                param_name = param_name.strip()
                value = value.strip()
                # Map 'time window' to 'tw' in the dictionary
                if param_name == 'time window':
                    param_name = 'tw'
                params_dict[current_participant][param_name] = float(value)
    
    return params_dict

def generate_predictions_df(input_csv: str, params_file: str, output_csv: str):
    """Generate predictions and create new CSV with prediction columns"""
    # Load original data
    original_df = pd.read_csv(input_csv)
    
    # Load parameters
    params_dict = parse_parameter_file(params_file)
    
    # Initialize model
    model = PT_TW_Model()
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
                g=p['gamma'],
                tw=p['tw']
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


def _format_participant_summary(participant: int, best_result: dict) -> List[str]:
    """Format summary lines for a participant's best parameter set."""
    params: Parameters = best_result['params']
    error: float = best_result['error']

    return [
        f"Participant {participant}:",
        f"Alpha: {params.a:.4f}",
        f"Beta: {params.b:.4f}",
        f"Lambda: {params.l:.4f}",
        f"Gamma: {params.g:.4f}",
        f"Time Window: {params.tw:.2f}",
        f"Mean Proportional Error: {error:.6f}",
        ""
    ]

def main():
    # Initialize model
    model = PT_TW_Model()
    
    # Load participant data
    data = load_dataset()
    model.data = data
    
    # Load parameter sets
    parameter_sets = load_parameter_sets()
    
    # Verify Subject column exists
    if 'Subject' not in parameter_sets.columns:
        print("Error: 'Subject' column not found in parameter sets file")
        sys.exit(1)
    
    # Optional: configure MPE from environment (defaults already set in __init__)
    # Example to mirror Excel: export PTTW_MPE_DENOM=sold; export PTTW_MPE_DAY_START=1

    # Test matching parameter sets
    results = test_all_participants(model, parameter_sets)
    
    # Print summary
    print("\nTesting Results Summary:")
    print("-" * 50)

    summary_lines = [
        "Testing Results Summary (Mean Proportional Errors):",
        "-" * 50,
        ""
    ]

    missing_participants: List[int] = []

    for participant, participant_results in sorted(results.items()):
        if not participant_results:
            print(f"\nParticipant {participant}: No parameter sets tested")
            missing_participants.append(participant)
            continue

        best_result = min(participant_results, key=lambda x: x['error'])

        print(f"\nParticipant {participant}:")
        print(f"Best parameter set (index {best_result['set_index']}):")
        print(f"Alpha: {best_result['params'].a:.4f}")
        print(f"Beta: {best_result['params'].b:.4f}")
        print(f"Lambda: {best_result['params'].l:.4f}")
        print(f"Gamma: {best_result['params'].g:.4f}")
        print(f"Time Window: {best_result['params'].tw}")
        print(f"Mean proportional error: {best_result['error']:.4f}")

        summary_lines.extend(_format_participant_summary(participant, best_result))

    # Add these lines at the end of main()
    input_csv = "/Users/brianwang/python-projs/GlassLab/PT_TW_DD PREDICTIONS.csv"
    params_file = "/Users/brianwang/python-projs/GlassLab/data/Best PTTW Fits.txt"
    output_csv = "/Users/brianwang/python-projs/GlassLab/PT_TW_WITH_PREDICTIONS.csv"
    
    generate_predictions_df(input_csv, params_file, output_csv)

    if summary_lines:
        output_path = Path(__file__).resolve().parents[1] / "output_files" / "PT TW Results.txt"
        output_text = "\n".join(summary_lines).rstrip() + "\n"
        output_path.write_text(output_text, encoding="utf-8")
        print(f"\nSaved summary to {output_path}")

    if missing_participants:
        print("\nParticipants without matching parameter sets:")
        for participant in missing_participants:
            print(f"  - {participant}")

    # Optional per-participant debug via env var
    dbg_pid = os.getenv("PTTW_DEBUG_PARTICIPANT")
    if dbg_pid:
        try:
            pid = int(dbg_pid)
            # Use the first param set available for this participant
            part_rows = parameter_sets[parameter_sets['Subject'] == pid]
            if not part_rows.empty:
                r0 = part_rows.iloc[0]
                dbg_params = Parameters(
                    a=float(r0['alphaNew']),
                    b=float(r0['betaNew']),
                    g=float(r0['gammaNew']),
                    l=float(r0['lambdaNew']),
                    tw=float(r0['twNew'])
                )
                outp = model.debug_participant_days(pid, dbg_params)
                print(f"Debug CSV for participant {pid}: {outp}")
            else:
                print(f"No parameter row found for participant {pid} to generate debug CSV.")
        except Exception as e:
            print(f"Failed to generate debug CSV for participant env '{dbg_pid}': {e}")

if __name__ == "__main__":
    main()



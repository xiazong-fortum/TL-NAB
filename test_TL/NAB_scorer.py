import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import math
import json
from collections import namedtuple

AnomalyPoint = namedtuple("AnomalyPoint", ["timestamp", "anomaly_score", "sweep_score", "window_name"])

class NABScorer:
    def __init__(self, probation_percent=0.15, profiles: Dict = None, threshold=None):
        """
        Initialize NAB scorer
        
        Args:
            profiles: Scoring configuration file, default is NAB's original three configurations
        """
        self.probation_percent = probation_percent
        self.threshold = threshold
        self.profiles = profiles or {
            "standard": {
                "tpWeight": 1.0,
                "fnWeight": 1.0, 
                "fpWeight": 0.11,
                "tnWeight": 1.0
            },
            "reward_low_FP_rate": {
                "tpWeight": 1.0,
                "fnWeight": 1.0,
                "fpWeight": 0.22, 
                "tnWeight": 1.0
            },
            "reward_low_FN_rate": {
                "tpWeight": 1.0,
                "fnWeight": 2.0,
                "fpWeight": 0.11,
                "tnWeight": 1.0
            }
        }
        
        self.windows = []
        
        
        
    def sigmoid(self, x):
        """Standard sigmoid function"""
        return 1 / (1 + math.exp(-x))

    def scaled_sigmoid(self, relative_position):
        """
        Return scaled sigmoid value based on the relative position in the labeled window
        
        Relative position explanation:
        -1.0: Leftmost edge of the anomaly window, S = 2*sigmoid(5) - 1.0 = 0.98661
        -0.5: Middle of the anomaly window, S = 2*sigmoid(0.5*5) - 1.0 = 0.84828
        0.0:  Right edge of the anomaly window, S = 2*sigmoid(0) - 1 = 0.0
        >0:   False positive outside the window, increasing penalty with distance
        """
        if relative_position > 3.0:
            # False positive far from the window
            val = -1.0
        else:
            val = 2 * self.sigmoid(-5 * relative_position) - 1.0
        return val

    def convert_windows(self, 
                       raw_labels: List[str],
                       timestamps: List[datetime],
                       probation_percent: float = 0.15,
                       window_size: float = 0.10) -> List[Tuple[datetime, datetime]]:
        """
        Convert raw labels to window labels
        
        Args:
            raw_labels: List of raw label timestamps
            timestamps: List of data timestamps
            probation_percent: Probation period percentage (default 15%)
            window_size: Window size percentage (default 10%)
            
        Returns:
            Processed list of windows [(start1, end1), (start2, end2),...]
        """
        # Convert raw_labels to datetime
        label_times = [pd.to_datetime(t) for t in raw_labels]
        
        # Calculate dataset length
        length = len(timestamps)
        num_anomalies = len(label_times)
        
        # Calculate window length
        if num_anomalies:
            window_length = int(window_size * length / num_anomalies)
        else:
            assert False, "No anomalies found in the data!"
            
        # Generate initial windows
        windows = []
        for anomaly_time in label_times:
            # Find the index of the anomaly point
            anomaly_idx = min(range(len(timestamps)), 
                             key=lambda i: abs(timestamps[i] - anomaly_time))
            
            # Calculate front and back boundaries of the window
            front = max(anomaly_idx, 0)
            back = min(anomaly_idx + window_length, length-1)
            
            # Add window
            windows.append((timestamps[front], timestamps[back]))
        
        # Handle probation period
        if windows:
            probation_idx = min(
                math.floor(probation_percent * length),
                probation_percent * 5000
            )
            probation_time = timestamps[probation_idx]
            
            # Remove windows overlapping with the probation period
            if windows and (windows[0][0] - probation_time).total_seconds() < 0:
                windows.pop(0)
        
        # Merge overlapping windows
        i = 0
        while len(windows) - 1 > i:
            if (windows[i+1][0] - windows[i][1]).total_seconds() <= 0:
                # Merge windows
                windows[i] = (windows[i][0], windows[i+1][1])
                windows.pop(i+1)
            else:
                i += 1

        return windows

    def score(self,
             data_dict: Dict[str, pd.DataFrame],
             label_dict: Dict[str, List[str]],
             profile: str = "standard") -> Dict[str, float]:
        """
        Calculate NAB scores for multiple time series
        
        Args:
            data_dict: Data dictionary in the format:
                {
                    "data1.csv": DataFrame (only containing timestamp and anomaly_score columns),
                    "data2.csv": DataFrame(...)
                }
            label_dict: Label dictionary in the format:
                {
                    "data1.csv": ["2014-01-02 00:00:00", "2014-01-05 00:00:00"],
                    "data2.csv": ["2014-01-04 00:00:00"]
                }
            profile: Scoring configuration name
            
        Returns:
            Dictionary of scores for each data file
            
        Raises:
            ValueError: When data format is incorrect or essential labels are missing
        """
        final_scores = {}
        
        # Check validity of data and label dictionaries
        if not data_dict:
            raise ValueError("Data dictionary cannot be empty")
        if not label_dict:
            raise ValueError("Label dictionary cannot be empty")
            
        for filename, df in data_dict.items():
            # Check for required columns
            if 'timestamp' not in df.columns:
                raise ValueError(f"{filename} is missing the timestamp column")
            if 'anomaly_score' not in df.columns:
                raise ValueError(f"{filename} is missing the anomaly_score column")
                
            if filename not in label_dict:
                raise ValueError(f"{filename} is missing label data")
                
            # Check validity of labels
            labels = label_dict[filename]
            if not labels:
                raise ValueError(f"{filename} has no label data")
                
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Convert windows
            windows = self.convert_windows(
                raw_labels=labels,
                timestamps=timestamps
            )
            print('windows:',windows)
            
            # Check validity of windows
            if not windows:
                raise ValueError(f"{filename} has no valid anomaly windows")
            
                       
            # Calculate optimal threshold and normalized score
            normalized_score, threshold, confusion_matrix = self.calc_score_by_threshold(
                timestamps=timestamps,
                anomaly_scores=df['anomaly_score'].values,
                windows=windows,
                profile=profile
            )
            
            final_scores[filename] = {'score':normalized_score, 'threshold':threshold, 'windows':windows, 'confusion_matrix':confusion_matrix}
            print(f"Score for {filename}: {normalized_score:.2f} (threshold={threshold})")
                
        return final_scores

    def calc_score_by_threshold(self, 
                                timestamps, 
                                anomaly_scores, 
                                windows, 
                                profile: str = "standard"):
        """Calculate the NAB score using either a specified threshold or by optimizing."""
        # Prepare score data
        score_parts = {}
        
        # Get profile weights
        weights = self.profiles[profile]
        
        # Calculate sweep scores for each point
        sweep_scores = self.calc_sweep_score(timestamps, anomaly_scores, windows, profile)
        
        # Initialize counts
        tn = sum(1 if x.window_name is None else 0 for x in sweep_scores)
        fn = sum(1 if x.window_name is not None else 0 for x in sweep_scores)
        tp = 0
        fp = 0
        
        # Sort scores by anomaly score in descending order
        sweep_scores.sort(key=lambda x: x.anomaly_score, reverse=True)
        scores_by_threshold = []
        cur_threshold = 1.0
        
        for point in sweep_scores:
            if point.anomaly_score != cur_threshold:
                # Save the current threshold's score
                cur_score = sum(score_parts.values())
                scores_by_threshold.append({
                    'threshold': cur_threshold,
                    'score': cur_score,
                    'confusion_matrix': {
                        'tp': tp,
                        'tn': tn,
                        'fp': fp,
                        'fn': fn
                    }
                })
                cur_threshold = point.anomaly_score
                
            # Update counts
            if point.window_name is not None:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
                
            # Update score parts
            if point.window_name is None:
                # False Positive score
                score_parts["fp"] = score_parts.get("fp", 0) + point.sweep_score * weights["fpWeight"]
            else:
                # True Positive score - only take the highest score per window
                score_parts[point.window_name] = max(
                    score_parts.get(point.window_name, float('-inf')),
                    point.sweep_score * weights["tpWeight"]
                )
        if self.threshold is not None:
            # Sort scores by threshold in descending order
            scores_by_threshold.sort(key=lambda x: x['threshold'])
            # Use the first score with threshold <= self.threshold
            best_score = next(score for score in scores_by_threshold if score['threshold'] >= self.threshold)
        else:
            # Find the best threshold from optimization
            best_score = max(scores_by_threshold, key=lambda x: x['score'])
        
        # Calculate the null score using the separate function
        null_score = self.calc_null_score(timestamps, windows, profile)
        
        # Calculate the perfect detector score
        perfect_score = self.get_perfect_score(windows, profile)
        
        # Ensure the denominator is not zero
        if perfect_score == null_score:
            normalized_score = 0
        else:
            normalized_score = 100 * (best_score['score'] - null_score) / (perfect_score - null_score)
        
        print(f"Threshold = {best_score['threshold']}, normalized score = {normalized_score:.2f}, score={best_score['score']}, (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        
        return normalized_score, best_score['threshold'], best_score['confusion_matrix']

    def get_probation_length(self, num_rows):
        """Determine the length of the probation period based on the number of rows."""
        return min(
            math.floor(self.probation_percent * num_rows),
            int(self.probation_percent * 5000)
        )

    def calc_sweep_score(self, timestamps: List, anomaly_scores: List[float], windows: List[Tuple], profile: str = "standard"):
        """Calculate sweep score for each point following the original NAB logic."""
        assert len(timestamps) == len(anomaly_scores), \
            "timestamps and anomaly_scores should not be different lengths!"

        # Create a copy of windows to avoid mutating the original list
        windows = list(windows)

        # Get the scoring weights from the profile
        weights = self.profiles.get(profile, self.profiles["standard"])

        # Prepare the final list of anomaly points to be returned
        anomaly_list = []

        # Configurable constants
        max_tp = self.scaled_sigmoid(-1.0)
        probation_length = self.get_probation_length(len(timestamps))

        # Variables updated during iteration
        cur_window = None
        cur_window_width = None
        cur_window_right_idx = None
        prev_window_width = None
        prev_window_right_idx = None

        # Track whether the current window has been credited with a TP
        # window_credited = False

        for i, (timestamp, score) in enumerate(zip(timestamps, anomaly_scores)):
            # Initialize scores
            unweighted_score = None
            weighted_score = None

            # Enter a new window if the current timestamp matches the start of the next window
            if windows and timestamp == windows[0][0]:
                cur_window = windows.pop(0)
                cur_window_right_idx = np.searchsorted(timestamps, cur_window[1], side='left')
                cur_window_width = float(cur_window_right_idx - i + 1)
                # window_credited = False  # Reset window credit flag for the new window

            # Calculate scores based on window status
            if cur_window is not None:
                # Within a window, calculate true positive score
                # if not window_credited and score > 0:
                position_in_window = -(cur_window_right_idx - i + 1) / cur_window_width
                unweighted_score = self.scaled_sigmoid(position_in_window)
                # weighted_score = unweighted_score * weights["tpWeight"] / max_tp
                weighted_score = unweighted_score * weights["tpWeight"]
                    # window_credited = True  # Mark this window as credited with a TP
                # else:
                    # weighted_score = 0.0  # Subsequent detections within the same window are not credited as TPs
            else:
                # Outside any window, calculate false positive score
                if prev_window_right_idx is None:
                    # No preceding window, use default score
                    unweighted_score = -1.0
                else:
                    # Calculate position outside of the previous window
                    position_past_window = abs(prev_window_right_idx - i) / float(prev_window_width - 1)
                    unweighted_score = self.scaled_sigmoid(position_past_window)

                weighted_score = unweighted_score * weights["fpWeight"]

            # Assign window name based on probation period
            if i >= probation_length:
                point_window_name = cur_window
            else:
                point_window_name = "probationary"

            # Create an AnomalyPoint instance and append it to the anomaly list
            point = AnomalyPoint(timestamp, score, weighted_score, point_window_name)
            anomaly_list.append(point)

            # Exit window if the current timestamp matches the window's right edge
            if cur_window is not None and timestamp == cur_window[1]:
                prev_window_right_idx = i
                prev_window_width = cur_window_width
                cur_window = None
                cur_window_width = None
                cur_window_right_idx = None

        return anomaly_list

    def calc_null_score(self, timestamps, windows, profile: str = "standard"):
        """
        Calculate the baseline score of a null detector that outputs a constant anomaly score of 0.5.
        
        Args:
            timestamps: List of datetime objects representing the time points.
            windows: List of tuples representing labeled anomaly windows.
            profile: The profile to use for scoring (e.g., "standard").
        
        Returns:
            The null score calculated based on the given profile.
        """
        # Get profile weights
        weights = self.profiles[profile]
        
        # Generate a constant anomaly score of 0.5 for all timestamps
        constant_scores = [0.5] * len(timestamps)
        
        # Calculate sweep scores for the constant anomaly score
        null_sweep_scores = self.calc_sweep_score(timestamps, constant_scores, windows, profile)
        
        # Calculate the null score
        null_score_parts = {}
        for point in null_sweep_scores:
            if point.window_name is None:
                # False Positive score
                null_score_parts["fp"] = null_score_parts.get("fp", 0) + point.sweep_score * weights["fpWeight"]
            else:
                # True Positive score - only take the highest score per window
                null_score_parts[point.window_name] = max(
                    null_score_parts.get(point.window_name, float('-inf')),
                    point.sweep_score * weights["tpWeight"]
                )
        null_score = sum(null_score_parts.values())
        
        return null_score

    def get_perfect_score(self, windows, profile="standard"):
        """Calculate the score of a perfect detector"""
        weights = self.profiles[profile]
        
        if not windows:
            return 1.0
            
        # Calculate maximum TP score
        max_tp_score = self.scaled_sigmoid(-1.0)  # Highest score at the start of the window
        
        # Calculate the ideal score for each window
        perfect_score = len(windows) * max_tp_score * weights["tpWeight"]
        
        # print(f"Perfect score calculation: no. of windows={len(windows)}, max_tp={max_tp_score}, weight={weights['tpWeight']}")
        
        return perfect_score  # Ensure score is not zero


######################## Example data generation ##############################
# start_date = pd.to_datetime('2014-01-01')
# dates = pd.date_range(start=start_date, periods=1000, freq='5min')
# anomaly_scores = np.random.rand(1000) * 0

# # Set obvious anomalies at certain positions
# anomaly_indices = [200, 300, 500, 700]  # Assume these positions are anomalies
# for idx in anomaly_indices:
#     anomaly_scores[idx] = 1  # Set high anomaly score

# data1 = pd.DataFrame({
#     'timestamp': dates,
#     'anomaly_score': anomaly_scores
# })
# # Save data to CSV file
# data1.to_csv('data1.csv', index=False)
# Labels should correspond to times when anomalies occurred
# label_dict = {
#     "data1.csv": [str(dates[idx]) for idx in anomaly_indices]
# }
# print(label_dict)


def main():
    # Load data
    data1 = pd.read_csv('./test_TL/results/inter_leakage_lstm_TL_u1.csv')
    # data2 = pd.read_csv('./test_TL/results/pump_failure_lstm_TL_u1.csv')

    label_dict = json.load(open('./test_TL/shf_labels.json'))

    data_dict = {
        "pump_failure.csv": data1,
        # "inter_leakage.csv": data2,
    }

    # Initialize scorer
    scorer = NABScorer(threshold=0.8)

    profiles = ["standard"]
    scores = []

    # Calculate scores
    for p in profiles:
        scores.append(scorer.score(data_dict, label_dict, profile=p))
    print('Final scores', scores)

if __name__ == "__main__":
    main()
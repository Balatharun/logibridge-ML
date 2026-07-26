"""
LogiEdge Preprocessing Pipeline
--------------------------------
Sequence: filtering -> windowed feature extraction -> normalisation

Feature vector (6 values, in this fixed order):
    0. temp_mean            (deg C)
    1. temp_std             (deg C)
    2. temp_rate_of_change  (deg C / min)
    3. vibration_rms        (g)
    4. vibration_peak       (g)
    5. vibration_kurtosis   (unitless)

Window = 30 seconds, step = 10 seconds (sliding window, overlapping).
Normalisation statistics are computed ONCE from 10 minutes of clean
Normal-class data and persisted to training_stats.npy. They must be
loaded at inference/runtime rather than recomputed on live data.
"""

import os
import numpy as np
from scipy import stats as scipy_stats

FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_rate_of_change",
    "vibration_rms", "vibration_peak", "vibration_kurtosis",
]

WINDOW_SECONDS = 30
STEP_SECONDS = 10
MOVING_AVG_SAMPLES = 5

STATS_PATH = os.path.join(os.path.dirname(__file__), "training_stats.npy")


def moving_average(series: np.ndarray, window: int = MOVING_AVG_SAMPLES) -> np.ndarray:
    """Simple causal moving average filter."""
    if len(series) < window:
        return series.astype(float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(series, kernel, mode="valid")
    # pad the front so output length matches input length
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def extract_features(temp_window: np.ndarray, temp_timestamps: np.ndarray,
                      vib_window: np.ndarray) -> np.ndarray:
    """
    temp_window       : 1-D array of temperature readings inside the 30 s window (post filter)
    temp_timestamps    : 1-D array of second-offsets aligned with temp_window (for rate-of-change)
    vib_window         : 1-D array of vibration RMS readings inside the 30 s window (post filter)
    Returns a 6-value feature vector, order = FEATURE_NAMES.
    """
    temp_mean = float(np.mean(temp_window))
    temp_std = float(np.std(temp_window))

    if len(temp_window) >= 2 and (temp_timestamps[-1] - temp_timestamps[0]) > 0:
        slope_per_sec = (temp_window[-1] - temp_window[0]) / (temp_timestamps[-1] - temp_timestamps[0])
        temp_rate = slope_per_sec * 60.0  # deg C per minute
    else:
        temp_rate = 0.0

    vib_rms = float(np.sqrt(np.mean(np.square(vib_window)))) if len(vib_window) else 0.0
    vib_peak = float(np.max(vib_window)) if len(vib_window) else 0.0
    if len(vib_window) >= 4:
        vib_kurtosis = float(scipy_stats.kurtosis(vib_window, fisher=True, bias=False))
    else:
        vib_kurtosis = 0.0

    return np.array([temp_mean, temp_std, temp_rate, vib_rms, vib_peak, vib_kurtosis],
                     dtype=np.float64)


def compute_and_save_stats(feature_matrix: np.ndarray, path: str = STATS_PATH) -> dict:
    """
    feature_matrix: (N, 6) array of feature vectors extracted from 10 minutes
                     of clean Normal-class data.
    Saves {"mean": (6,), "std": (6,)} to a .npy file (allow_pickle=True).
    """
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1e-6  # guard against divide-by-zero on a constant feature
    stats = {"mean": mean, "std": std}
    np.save(path, stats, allow_pickle=True)
    return stats


def load_stats(path: str = STATS_PATH) -> dict:
    stats = np.load(path, allow_pickle=True).item()
    return stats


def normalise(feature_vector: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score normalisation using persisted training statistics only."""
    return (feature_vector - stats["mean"]) / stats["std"]


class SlidingWindowBuffer:
    """
    Maintains rolling temperature and vibration buffers and yields a new
    feature vector every STEP_SECONDS once a full WINDOW_SECONDS of data
    is available. Used both by the offline dataset generator and by the
    live inference service.
    """

    def __init__(self):
        self.temp_values = []
        self.temp_times = []
        self.vib_values = []
        self._last_emit_time = 0.0

    def add_temperature(self, t_offset: float, value: float):
        self.temp_values.append(value)
        self.temp_times.append(t_offset)
        self._trim(t_offset)

    def add_vibration(self, t_offset: float, value: float):
        self.vib_values.append(value)
        self._trim(t_offset)

    def _trim(self, t_offset: float):
        cutoff = t_offset - WINDOW_SECONDS
        while self.temp_times and self.temp_times[0] < cutoff:
            self.temp_times.pop(0)
            self.temp_values.pop(0)

    def ready(self, t_offset: float) -> bool:
        has_window = t_offset >= WINDOW_SECONDS
        due = (t_offset - self._last_emit_time) >= STEP_SECONDS
        return has_window and due and len(self.temp_values) > 0

    def emit(self, t_offset: float) -> np.ndarray:
        temp_arr = moving_average(np.array(self.temp_values))
        vib_arr = moving_average(np.array(self.vib_values[-len(self.temp_values):]
                                           if self.vib_values else [0.0]))
        features = extract_features(temp_arr, np.array(self.temp_times), vib_arr)
        self._last_emit_time = t_offset
        return features

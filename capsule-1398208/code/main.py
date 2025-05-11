import math
import os
import pandas as pd
import numpy as np
from scipy.signal import resample
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from datetime import datetime

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from detect_peaks import detect_peaks
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot

# Main program starts here
print('\nstart processing ...')

# File paths
bcg_file = '/Volumes/MySSD/Data Analytics Team 1/capsule-1398208/dataset/data/01/BCG/01_20231105_BCG.csv'
ecg_file = '/Volumes/MySSD/Data Analytics Team 1/capsule-1398208/dataset/data/01/Reference/RR/01_20231105_RR.csv'

# Load BCG data
if bcg_file.endswith(".csv"):
    bcg_fileName = os.path.join(bcg_file)
    if os.path.exists(bcg_fileName) and os.stat(bcg_fileName).st_size != 0:
        bcg_rawData = pd.read_csv(bcg_fileName, sep=",", header=0).values
        print(f"BCG raw data shape: {bcg_rawData.shape}")
        if bcg_rawData.size > 0:
            bcg_data_stream = bcg_rawData[:, 0].astype(float)  # BCG signal
            # Generate BCG timestamps based on first timestamp and 140 Hz
            bcg_start_time_ms = bcg_rawData[0, 1]  # First timestamp in milliseconds
            fs_bcg = 140  # Sampling frequency
            time_step_ms = 1000 / fs_bcg  # Interval in milliseconds
            bcg_time_ms = bcg_start_time_ms + np.arange(len(bcg_data_stream)) * time_step_ms
            # Convert BCG timestamps to datetime
            bcg_time = pd.to_datetime(bcg_time_ms, unit='ms')
            print(f"BCG data length: {len(bcg_data_stream)}, First timestamp: {bcg_time[0]}, Last timestamp: {bcg_time[-1]}")
        else:
            print("Error: BCG data is empty after loading.")
            exit(1)
    else:
        print(f"Error: BCG file {bcg_fileName} not found or empty.")
        exit(1)

# Load ECG data
if ecg_file.endswith(".csv"):
    ecg_fileName = os.path.join(ecg_file)
    if os.path.exists(ecg_fileName) and os.stat(ecg_fileName).st_size != 0:
        ecg_rawData = pd.read_csv(ecg_fileName, sep=",", header=0).values
        print(f"ECG raw data shape: {ecg_rawData.shape}")
        if ecg_rawData.size > 0:
            ecg_time = pd.to_datetime(ecg_rawData[:, 0])  # Already in datetime format
            ecg_hr = ecg_rawData[:, 1].astype(float)  # Heart Rate
            print(f"ECG data length: {len(ecg_hr)}, First timestamp: {ecg_time[0]}, Last timestamp: {ecg_time[-1]}")
        else:
            print("Error: ECG data is empty after loading.")
            exit(1)
    else:
        print(f"Error: ECG file {ecg_fileName} not found or empty.")
        exit(1)

# Sync BCG and ECG data based on timestamps
def sync_data(bcg_time, bcg_data, ecg_time, ecg_hr):
    print(f"BCG range: {bcg_time[0]} to {bcg_time[-1]}")
    print(f"ECG range: {ecg_time[0]} to {ecg_time[-1]}")
    start_time = max(bcg_time[0], ecg_time[0])
    end_time = min(bcg_time[-1], ecg_time[-1])
    print(f"Sync range: {start_time} to {end_time}")
    if start_time > end_time:
        print("Error: No temporal overlap between BCG and ECG data.")
        print(f"BCG ends at {bcg_time[-1]}")
        print(f"ECG starts at {ecg_time[0]}")
        exit(1)
    # Filter data within the overlapping range
    bcg_mask = (bcg_time >= start_time) & (bcg_time <= end_time)
    ecg_mask = (ecg_time >= start_time) & (ecg_time <= end_time)
    bcg_time_synced = bcg_time[bcg_mask]
    bcg_data_synced = bcg_data[bcg_mask]
    ecg_time_synced = ecg_time[ecg_mask]
    ecg_hr_synced = ecg_hr[ecg_mask]
    print(f"Synced BCG data length: {len(bcg_data_synced)}, Synced ECG data length: {len(ecg_hr_synced)}")
    # Convert synced timestamps back to milliseconds for downstream processing
    bcg_time_synced_ms = bcg_time_synced.astype('int64') // 10**6  # Convert to milliseconds
    ecg_time_synced_ms = ecg_time_synced.astype('int64') // 10**6
    return bcg_time_synced_ms, bcg_data_synced, ecg_time_synced_ms, ecg_hr_synced

bcg_utc_time, bcg_data_stream, ecg_utc_time, ecg_hr = sync_data(bcg_time, bcg_data_stream, ecg_time, ecg_hr)

# Resample BCG data from 140 Hz to 50 Hz
original_fs = 140  # Original sampling rate
target_fs = 50     # Target sampling rate
num_samples = int(len(bcg_data_stream) * (target_fs / original_fs))
if len(bcg_data_stream) > 0:
    resampled_bcg_data = resample(bcg_data_stream, num_samples)
    # Update utc_time for resampled data (linear interpolation)
    resampled_utc_time = np.interp(np.linspace(0, len(bcg_utc_time)-1, num_samples), 
                                  np.arange(len(bcg_utc_time)), bcg_utc_time)
else:
    print("Error: No data to resample.")
    exit(1)

# Run motion artifact detection
start_point, end_point, window_shift, fs = 0, 500, 500, 50
resampled_bcg_data, resampled_utc_time = detect_patterns(start_point, end_point, window_shift, 
                                                        resampled_bcg_data, resampled_utc_time, plot=1)

# BCG signal extraction
movement = band_pass_filtering(resampled_bcg_data, fs, "bcg")

# Respiratory signal extraction
breathing = band_pass_filtering(resampled_bcg_data, fs, "breath")
breathing = remove_nonLinear_trend(breathing, 3)
breathing = savgol_filter(breathing, 11, 3)

# Wavelet transformation
w = modwt(movement, 'bior3.9', 4)
dc = modwtmra(w, 'bior3.9')
wavelet_cycle = dc[4]

# Vital Signs estimation over windows
t1, t2, window_length, window_shift = 0, 500, 500, 500
hop_size = math.floor((window_length - 1) / 2)
limit = int(math.floor(breathing.size / window_shift))

# Heart Rate using vitals with compute_rate
mpd = 1  # Minimum peak distance
beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, resampled_utc_time, mpd, plot=0)
print('\nHeart Rate Information')
print('Minimum pulse : ', np.around(np.min(beats), 2) if len(beats) > 0 else 'N/A')
print('Maximum pulse : ', np.around(np.max(beats), 2) if len(beats) > 0 else 'N/A')
print('Average pulse : ', np.around(np.mean(beats), 2) if len(beats) > 0 else 'N/A')

# Breathing Rate using compute_vitals
beats_breath = vitals(t1, t2, window_shift, limit, breathing, resampled_utc_time, mpd, plot=0)
print('\nRespiratory Rate Information')
print('Minimum breathing : ', np.around(np.min(beats_breath), 2))
print('Maximum breathing : ', np.around(np.max(beats_breath), 2))
print('Average breathing : ', np.around(np.mean(beats_breath), 2))

# Apnea events detection
thresh = 0.3
events = apnea_events(breathing, resampled_utc_time, thresh=thresh)

# Plot Vitals Example using data_subplot
t1, t2 = 2500, 5000  # Window from 50 to 100 seconds at 50 Hz
data_subplot(resampled_bcg_data, movement, breathing, wavelet_cycle, t1, t2)

# ----- J-Peak Detection Attempt -----
def naive_j_peak_detection(bcg_signal, r_peak_indices, search_window_ms=100, fs=50):
    """
    A very basic attempt to find J-peaks in a BCG signal.
    This is highly simplified and will likely need significant tuning.

    Args:
        bcg_signal: The filtered BCG signal.
        r_peak_indices: Indices of the R-peaks (or BCG equivalent) in the BCG signal.
        search_window_ms:  How many milliseconds after the R-peak to look for a J-peak.
        fs: Sampling rate of the BCG signal.

    Returns:
        A list of indices where J-peaks are tentatively located.
    """

    j_peak_indices = []
    search_window_samples = int(search_window_ms * (fs / 1000))  # Window in samples

    for r_index in r_peak_indices:
        start_search = r_index + 1  # Don't search at the R-peak itself
        end_search = min(len(bcg_signal), r_index + search_window_samples)

        if start_search < end_search:
            window = bcg_signal[start_search:end_search]
            if len(window) > 0:
              j_peak_local_index = np.argmax(window)  # Index relative to the window
              j_peak_index = start_search + j_peak_local_index
              j_peak_indices.append(j_peak_index)

    return j_peak_indices

#  1. Find R-peak proxies in BCG (using your existing peak detection)
#     -  This is CRITICAL:  BCG "R-peaks" are NOT the same as ECG R-peaks.
#     -  We're finding the BCG wave that corresponds *roughly* to the heartbeat.
bcg_r_peak_indices = detect_peaks(wavelet_cycle, mpd=20)  # Adjust 'mpd'!

#  2. Detect J-peaks relative to those BCG "R-peaks"
bcg_j_peak_indices = naive_j_peak_detection(wavelet_cycle, bcg_r_peak_indices, search_window_ms=80, fs=fs) # Tune window!

# ----- End J-Peak Detection -----

# Output the detected J-peak indices and times
if bcg_j_peak_indices:
    j_peak_times = resampled_utc_time[bcg_j_peak_indices]
    print("\nDetected J-peak Indices (in resampled BCG signal):", bcg_j_peak_indices)
    print("Detected J-peak Times:", j_peak_times)
else:
    print("\nNo J-peaks detected.")

# Evaluation Metrics: Compare estimated HR with reference ECG HR (if synced data exists)
if len(ecg_hr) > 0:
    # Interpolate ECG HR to match the time points of beats
    beats_time = resampled_utc_time[::window_shift][:len(beats)]
    ecg_hr_interpolated = np.interp(beats_time, ecg_utc_time, ecg_hr)

    # Compute MAE and RMSE
    if len(beats) > 0 and len(ecg_hr_interpolated) > 0:
        mae = np.mean(np.abs(beats - ecg_hr_interpolated))
        rmse = np.sqrt(np.mean((beats - ecg_hr_interpolated)**2))
        print('\nEvaluation Metrics')
        print('Mean Absolute Error (MAE): ', np.around(mae, 2))
        print('Root Mean Square Error (RMSE): ', np.around(rmse, 2))

        # Pearson Correlation
        corr, _ = pearsonr(beats, ecg_hr_interpolated)
        print('Pearson Correlation Coefficient: ', np.around(corr, 2))

        # Bland-Altman Plot
        mean_hr = (beats + ecg_hr_interpolated) / 2
        diff_hr = beats - ecg_hr_interpolated
        mean_diff = np.mean(diff_hr)
        std_diff = np.std(diff_hr)
        plt.figure(figsize=(8, 6))
        plt.scatter(mean_hr, diff_hr, c='blue', alpha=0.5)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
        plt.title('Bland-Altman Plot')
        plt.xlabel('Mean of Estimated and Reference HR')
        plt.ylabel('Difference (Estimated HR - Reference HR)')
        plt.savefig('bland_altman_plot.png')

        # Pearson Correlation Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(ecg_hr_interpolated, beats, c='blue', alpha=0.5)
        plt.plot([min(ecg_hr_interpolated), max(ecg_hr_interpolated)], 
                 [min(ecg_hr_interpolated), max(ecg_hr_interpolated)], 'r--')
        plt.title(f'Pearson Correlation Plot (r = {corr:.2f})')
        plt.xlabel('Reference HR (ECG)')
        plt.ylabel('Estimated HR (BCG)')
        plt.savefig('pearson_correlation_plot.png')

        # Boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot([beats, ecg_hr_interpolated], tick_labels=['Estimated HR (BCG)', 'Reference HR (ECG)'])
        plt.title('Boxplot of Estimated and Reference HR')
        plt.ylabel('Heart Rate (bpm)')
        plt.savefig('boxplot_hr.png')
else:
    print("\nWarning: No ECG data for evaluation metrics due to no overlap.")

# ----- J-Peak Visualization (Illustrative) -----
plt.figure(figsize=(10, 5))
plt.plot(resampled_utc_time, wavelet_cycle, label='BCG Wavelet', color='blue')
plt.plot(resampled_utc_time[bcg_r_peak_indices], wavelet_cycle[bcg_r_peak_indices], 'ro', label='BCG "R-peaks"')
plt.plot(resampled_utc_time[bcg_j_peak_indices], wavelet_cycle[bcg_j_peak_indices], 'go', label='Detected J-peaks')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('BCG with Detected Peaks')
plt.legend()
plt.savefig('j_peak_visualization.png') # Changed from plt.show() to plt.savefig()

print('\nEnd processing ...')
s
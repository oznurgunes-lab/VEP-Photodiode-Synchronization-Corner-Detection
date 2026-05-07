"""
VEP System Validation & Photodiode Synchronization Analysis
Author: Öznur
Date: 2026
Description: This script validates stimulus timing using Corner Detection 
algorithms for both onsets and offsets in VEP recordings.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Change this path to your local .cnt or .ant file
FILE_PATH = 'path/to/your/data_file.cnt' 
TARGET_CHANNELS = ['O1', 'O2', 'Oz', 'Pz']
PHOTO_CH = '23'
REJECT_CRITERIA = dict(eeg=150e-6) # 150 µV rejection threshold [cite: 18]

def load_data(file_path):
    """Loads EEG data from .ant or .cnt formats."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True, verbose=False)
    except:
        raw = mne.io.read_raw_cnt(file_path, data_format='int32', preload=True, verbose=False)
    return raw

def detect_events_corner(raw, onset_thresh=0.20, offset_thresh=0.80):
    """
    Implements Back-tracking Corner Detection for precise timing.
    Low thresholds (e.g., 0.5%-5%) are precise for clean data[cite: 26].
    High thresholds (e.g., 20%) are stable for noisy data[cite: 29].
    """
    sfreq = raw.info['sfreq']
    photo_data = raw.get_data(picks=PHOTO_CH)[0]
    # Normalization
    p_norm = (photo_data - np.min(photo_data)) / (np.max(photo_data) - np.min(photo_data))
    
    ons, offs = [], []
    i = 1
    while i < len(p_norm) - int(0.2 * sfreq):
        # ONSET detection (Rising edge)
        if p_norm[i] >= 0.5 > p_norm[i-1]:
            idx = i
            while idx > 0 and p_norm[idx] > onset_thresh:
                idx -= 1
            ons.append(idx)
            i += int(0.4 * sfreq)
        # OFFSET detection (Falling edge)
        elif p_norm[i] <= 0.5 < p_norm[i-1]:
            idx = i
            while idx > 0 and p_norm[idx] < offset_thresh:
                idx -= 1
            offs.append(idx)
            i += int(0.4 * sfreq)
        else:
            i += 1
            
    return np.unique(ons), np.unique(offs)

def run_full_analysis(file_path):
    raw = load_data(file_path)
    if raw is None: return

    # Initial filtering [cite: 16, 17]
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    
    ons, offs = detect_events_corner(raw)
    
    # Create Events
    ev_ons = np.column_stack([ons + raw.first_samp, np.zeros_like(ons), np.ones_like(ons)]).astype(int)
    ev_offs = np.column_stack([offs + raw.first_samp, np.zeros_like(offs), np.full_like(offs, 2)]).astype(int)
    
    # Epoching
    # Note: Controlled environment results in minimal data loss [cite: 19]
    epochs_on = mne.Epochs(raw, ev_ons, event_id={'Onset': 1}, tmin=-0.1, tmax=0.4, 
                           baseline=(None, 0), picks=TARGET_CHANNELS, preload=True, 
                           reject=REJECT_CRITERIA, verbose=False, event_repeated='drop')
    
    epochs_off = mne.Epochs(raw, ev_offs, event_id={'Offset': 2}, tmin=-0.1, tmax=0.4, 
                            baseline=(None, 0), picks=TARGET_CHANNELS, preload=True, 
                            reject=REJECT_CRITERIA, verbose=False, event_repeated='drop')

    # Plotting results
    avg_ons = epochs_on.average()
    avg_offs = epochs_off.average()
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_ons.times * 1000, np.mean(avg_ons.data, axis=0) * 1e6, label='Onset Average')
    plt.plot(avg_offs.times * 1000, np.mean(avg_offs.data, axis=0) * 1e6, label='Offset Average', linestyle='--')
    plt.axvline(0, color='black', alpha=0.5, label='Stimulus Trigger')
    plt.title('VEP System Validation: Onset vs Offset Detection')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_full_analysis(FILE_PATH)

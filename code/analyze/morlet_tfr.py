from pathlib import Path
import mne
from mne_bids import BIDSPath
import numpy as np

# Parameters
subjects = [2]  # Add as many subject IDs as you need
task = "craa"
desc = "Apply morlet wavlet transform."

base_dir = Path("..") / "data" / "derivatives"
in_folder = "epoch"
out_folder = "cwt"

# Define frequency bands of interest
conds = range(1, 22)
freqs_of_interest = [50, 30, 14, 8, 4, 2]
# n_cycles = 2  # TODO: how to best balance time and frequency resolution?

maxFreq = 50
numFreq = 96
DJ = 0.05  # Scale spacing (similar to MATLAB parameter)

freqs = maxFreq * 2 ** (-np.arange(numFreq) * DJ)
n_cycles = (1 / DJ) * (freqs / maxFreq)   # Dynamic n_cycles based on scale spacing
breakpoint()
# Find indices corresponding to freqOI
idxfreq = np.array([np.argmin(abs(freqs - f)) for f in freqs_of_interest])
bands = [slice(idxfreq[i], idxfreq[i+1]) for i in range(len(freqs_of_interest) - 1)]

batch_size = 200  # Number of trials per batch

for sub_id in subjects:
    sub_str = f"sub-{sub_id:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")
    in_p = base_dir / in_folder / sub_str / "eeg"
    in_file_cue = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-cue_epo.fif"
    in_file_target = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_type-target_epo.fif"
    cond_file = in_p / f"{sub_str}_task-{task}_proc-{in_folder}_conditions.npy"
    condition_labels = np.load(cond_file)
    
    bids_out = BIDSPath(
        subject=sub_str.split('-')[1],
        task=task,
        processing=out_folder,
        extension='.fif',
        datatype='eeg',
        root=base_dir / out_folder
    )
    out_file = bids_out.fpath
    out_file.parent.mkdir(parents=True, exist_ok=True) 

    print(f"Loading file: {in_file_cue}")
    epochs_cue = mne.read_epochs(in_file_cue, preload=True)
    epochs_target = mne.read_epochs(in_file_target, preload=True)

    for epochs_i, epochs in enumerate([epochs_cue, epochs_target]):
        event_type = 'cue' if epochs_i == 0 else 'target'
        print(f"Processing {event_type} epochs")
        total_trials = len(epochs)
        collected_band_amplitudes = []
        for cond_i in conds:
            print(f"Processing condition {cond_i}")
            batch_epochs = epochs[condition_labels == cond_i]

            power = batch_epochs.compute_tfr(
                method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=4
            )
            power_data = power.data  # (trials, channels, freqs, time)
            band_amplitude = np.zeros((power_data.shape[0], power_data.shape[1], len(bands), power_data.shape[3]))
            for i, band in enumerate(bands):
                band_amplitude[:, :, i, :] = np.mean(power_data[:, :, band, :], axis=2)  # Mean amplitude in band

            collected_band_amplitudes.append(band_amplitude)
        final_band_amplitude = np.stack(collected_band_amplitudes, axis=0)  # (condition, trial, channel, feature(band), time)
        
        np.save(out_file.parent / f"{out_file.stem}_type-{event_type}.npy", final_band_amplitude)
        print(f"Saved final band amplitude data")
        
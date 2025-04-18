from pathlib import Path
import mne
from mne_bids import BIDSPath
import numpy as np

# Parameters
def run(config, sub_i):
    task = config["task"]

    base_dir = Path(config["base_dir"])
    config_cwt = config["cwt"]
    in_dir = base_dir / config_cwt["input_folder"]
    out_dir = base_dir / config_cwt["output_folder"]
    
    # Define frequency bands of interest
    conds = range(1, config_cwt["num_condition"] + 1)
    freq_boundary = config_cwt["frequency_boundary"]

    maxFreq = config_cwt["frequency_boundary"][0]
    numFreq = config_cwt["num_frequency_bin"]
    DJ = config_cwt["scale_spacing_DJ"]  # Scale spacing (similar to MATLAB parameter)
    freqs = maxFreq * 2 ** (-np.arange(numFreq) * DJ)
    n_cycles = (1 / DJ) * (freqs / maxFreq)   # Dynamic n_cycles based on scale spacing
    
    # Find indices corresponding to freqOI
    idxfreq = np.array([np.argmin(abs(freqs - f)) for f in freq_boundary])
    bands = [slice(idxfreq[i], idxfreq[i+1]) for i in range(len(freq_boundary) - 1)]

    sub_str = f"sub-{sub_i:03d}"
    print(f"\n=== Processing subject: {sub_str} ===")
    
    for event_type in ["cue", "target"]:
        bids_in = BIDSPath(
            subject=sub_str.split('-')[1],
            task=task,
            processing=in_dir.name,
            datatype="eeg",
            root=in_dir,
            description=event_type
        )
        in_fpath = bids_in.fpath
        label_fpath = in_fpath.parent / (in_fpath.stem.split("_desc")[0] + '_conditions.npy')
        condition_labels = np.load(label_fpath)
        
        bids_out = BIDSPath(
            subject=sub_str.split('-')[1],
            task=task,
            processing=out_dir.name,
            datatype='eeg',
            root=out_dir,
            description=event_type
        )
        bids_out.fpath.parent.mkdir(parents=True, exist_ok=True)
        out_fpath = bids_out.fpath
        epochs = mne.read_epochs(in_fpath, preload=True)
       
        # 1) Compute TFR on each conditon
        # print(f"Processing {event_type} epochs")
        # total_trials = len(epochs)
        # collected_band_amplitudes = []
        # for cond_i in conds:
        #     print(f"Processing condition {cond_i}")
        #     batch_epochs = epochs[condition_labels == cond_i]

        #     power = batch_epochs.compute_tfr(
        #         method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=1
        #     )
        #     power_data = power.data  # (trials, channels, freqs, time)
        #     band_amplitude = np.zeros((power_data.shape[0], power_data.shape[1], len(bands), power_data.shape[3]))
        #     for i, band in enumerate(bands):
        #         band_amplitude[:, :, i, :] = np.mean(power_data[:, :, band, :], axis=2)  # Mean amplitude in band

        #     collected_band_amplitudes.append(band_amplitude)
        # final_band_amplitude = np.stack(collected_band_amplitudes, axis=0)  # (condition, trial, channel, feature(band), time)

        print(f"Processing {event_type} epochs")

        # 2) Compute TFR on the whole epoch set
        power = epochs.compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=-1
        )
        power_data = power.data  # Shape: (trials, channels, freqs, time)
        
        final_band_amplitude = []
        for cond_i in conds:
            print(f"Processing condition {cond_i}")
            condition_power = power_data[condition_labels == cond_i]  # Select trials for condition
            
            # Compute mean amplitude within each frequency band
            band_amplitude = np.zeros((condition_power.shape[0], condition_power.shape[1], len(bands), condition_power.shape[3]))
            for i, band in enumerate(bands):
                band_amplitude[:, :, i, :] = np.mean(condition_power[:, :, band, :], axis=2)  # Mean amplitude in band

            final_band_amplitude.append(band_amplitude)
            print(band_amplitude.shape)
            del condition_power, band_amplitude

        final_band_amplitude = np.stack(final_band_amplitude, axis=0)  # Shape: (condition, trial, channel, band, time)
        print(final_band_amplitude.shape)
        
        np.save(out_fpath.with_suffix(".npy"), final_band_amplitude)
        print(f"Saved final band amplitude data")
        del final_band_amplitude, power_data, power, epochs
        
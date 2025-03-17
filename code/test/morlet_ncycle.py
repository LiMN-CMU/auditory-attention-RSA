import numpy as np
from matplotlib import pyplot as plt

from mne import Epochs, create_info
from mne.io import RawArray
from mne.time_frequency import AverageTFRArray, EpochsTFRArray, tfr_array_morlet

sfreq = 1000.0
ch_names = ["SIM0001", "SIM0002"]
ch_types = ["grad", "grad"]
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = 1024  # Just over 1 second epochs
n_epochs = 40
seed = 42
rng = np.random.RandomState(seed)
data = rng.randn(len(ch_names), n_times * n_epochs + 200)  # buffer

freqs = np.arange(5.0, 100.0, 3.0)
vmin, vmax = -3.0, 3.0  # Define our color limits.

# Add a 50 Hz sinusoidal burst to the noise and ramp it.
t = np.arange(n_times, dtype=np.float64) / sfreq
signal = np.sin(np.pi * 2.0 * 50.0 * t)  # 50 Hz sinusoid signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.0  # hard windowing
on_time = np.logical_and(t >= 0.45, t <= 0.55)
signal[on_time] *= np.hanning(on_time.sum())  # ramping
data[:, 100:-100] += np.tile(signal, n_epochs)  # add signal

raw = RawArray(data, info)
events = np.zeros((n_epochs, 3), dtype=int)
events[:, 0] = np.arange(n_epochs) * n_times
epochs = Epochs(
    raw,
    events,
    dict(sin50hz=0),
    tmin=0,
    tmax=n_times / sfreq,
    reject=dict(grad=4000),
    baseline=None,
)

# epochs.average().plot()

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
all_n_cycles = [freqs / 5.0, freqs / 3.0, freqs / 2.0]
for n_cycles, ax in zip(all_n_cycles, axs):
    power = epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True
    )
    power.plot(
        [0],
        baseline=(0.0, 0.1),
        mode="mean",
        vlim=(vmin, vmax),
        axes=ax,
        show=False,
        colorbar=False,
    )
    n_cycles = "scaled by freqs" if not isinstance(n_cycles, int) else n_cycles
    ax.set_title(f"Sim: Using Morlet wavelet, n_cycles = {n_cycles}")
plt.show()
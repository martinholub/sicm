import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

class LowPassButter(object):
    """
    Q: Should I not apply the filter in frequency domain?

    Parameters
    -------------
    fs: float
        Sampling frequency, decrease for smoother lines. Watch out for extension
        of inital artifact period. [default = 10]
    cutoff: float
        Decrease for smoother fits. Watch out for extension of initial artifact
        period. [default = 10]
    order: int
        Increasing order leads to extension of initial artifact period.

    """
    def __init__(self):
        pass

    def butter_lowpass(self, cutoff, fs, order = 5):
        """Design Butterworth lowpass filter"""
        nyq = .5 * fs
        normal_cutoff = cutoff / nyq
        # Design filter
        b, a = butter(order, normal_cutoff, btype = "low", analog = False)
        # self.plot_frequency_response(a, b, fs, cutoff)
        return b, a

    def filter(self, data, cutoff, fs, order):
        """Apply filter"""
        b, a = self.butter_lowpass(cutoff, fs, order = order)
        y = lfilter(b, a, data)
        return y

    def plot_frequency_response(self, a, b, fs, cutoff):
        """Plot the frequency response."""
        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))

        w, h = freqz(b, a, worN=8000)
        ax.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        ax.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        ax.axvline(cutoff, color='k')
        ax.set_xlim(0, 0.5*fs)
        ax.set_title("Lowpass Filter Frequency Response")
        ax.set_xlabel('Frequency [Hz]')
        ax.grid()

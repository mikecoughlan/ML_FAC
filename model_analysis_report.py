import os
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
# Assuming PostProcessingAnalysis class is already defined
# from your_module import PostProcessingAnalysis
import numpy as np
import pandas as pd
from numba import jit
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings('ignore')


class PostProcessingAnalysis:
    """
    Comprehensive post-processing analysis system for geomagnetic model predictions.

    Supports both deterministic predictions (mean only) and probabilistic predictions
    (mean + standard deviation).

    Includes 15 analysis methods:
    - 8 deterministic analyses
    - 7 uncertainty-aware analyses (require std data)
    """

    def __init__(self, results_dict, time_range=None, timestamp_format=None, has_uncertainty=None):
        """
        Initialize the PostProcessingAnalysis object.

        Parameters:
        -----------
        results_dict : dict
            Dictionary with timestamps as keys and dictionaries as values.
            Each inner dictionary must contain:
                - 'ampere': DataFrame with AMPERE observations
                - 'predicted': DataFrame with model predictions
                - 'std': DataFrame with prediction uncertainties (optional)

            Example:
            {
                '2015-03-17 00:00:00': {
                    'ampere': df_ampere,
                    'predicted': df_predicted,
                    'std': df_std  # optional
                },
                ...
            }

        time_range : tuple or None
            Optional (start_time, end_time) to filter data

        timestamp_format : str or None
            Format string for parsing timestamps (e.g., '%Y-%m-%d %H:%M:%S')

        has_uncertainty : bool or None
            Whether uncertainty data is available. If None, auto-detects.
        """
        self.results_dict = results_dict
        self.timestamp_format = timestamp_format

        # Extract and sort timestamps
        self.timestamps = sorted(results_dict.keys())

        # Apply time filtering if specified
        if time_range is not None:
            start_time, end_time = time_range
            if isinstance(start_time, str):
                start_time = self.parse_timestamp(start_time, timestamp_format)
            if isinstance(end_time, str):
                end_time = self.parse_timestamp(end_time, timestamp_format)

            filtered_timestamps = []
            for ts in self.timestamps:
                parsed_ts = self.parse_timestamp(ts, timestamp_format)
                if start_time <= parsed_ts <= end_time:
                    filtered_timestamps.append(ts)

            self.timestamps = filtered_timestamps

        if len(self.timestamps) == 0:
            raise ValueError("No data found in specified time range")

        # Get dimensions from first dataframe
        first_data = results_dict[self.timestamps[0]]
        first_df = first_data['ampere']

        self.n_times = len(self.timestamps)
        self.n_lats = len(first_df)
        self.n_mlts = len(first_df.columns)

        # Auto-detect uncertainty availability
        if has_uncertainty is None:
            self.has_uncertainty = 'std' in first_data and first_data['std'] is not None
        else:
            self.has_uncertainty = has_uncertainty

        # Pre-stack data into 3D arrays for efficient access
        print("Preprocessing data into 3D arrays...")
        self.ampere_stack = self._stack_data('ampere')
        self.predicted_stack = self._stack_data('predicted')

        if self.has_uncertainty:
            self.std_stack = self._stack_data('std')
        else:
            self.std_stack = None

        print(f"Loaded {self.n_times} timesteps")
        print(f"Grid size: {self.n_lats} latitudes × {self.n_mlts} MLTs")
        print(f"Uncertainty available: {self.has_uncertainty}")

    def _stack_data(self, key):
        """
        Stack dataframes into a 3D numpy array (time, lat, mlt).

        Parameters:
        -----------
        key : str
            Key to extract from results_dict ('ampere', 'predicted', or 'std')

        Returns:
        --------
        stacked : ndarray
            3D array of shape (n_times, n_lats, n_mlts)
        """
        stacked = np.empty((self.n_times, self.n_lats, self.n_mlts))

        for i, ts in enumerate(self.timestamps):
            df = self.results_dict[ts][key]
            stacked[i, :, :] = df.values

        return stacked

    @staticmethod
    def parse_timestamp(timestamp, fmt=None):
        """
        Parse timestamp string to datetime object.

        Parameters:
        -----------
        timestamp : str or datetime
            Timestamp to parse
        fmt : str or None
            Format string for parsing

        Returns:
        --------
        parsed : datetime
            Parsed datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp

        if fmt is not None:
            return datetime.strptime(timestamp, fmt)

        # Try common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y%m%d_%H%M%S']:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue

        # Last resort: let pandas try
        return pd.to_datetime(timestamp)

    def lat_to_colat(self, lat):
        """
        Convert latitude (degrees) to co-latitude index.
        Assumes latitude grid from 90° (index 0) to lower latitudes.

        Parameters:
        -----------
        lat : int or float
            Latitude in degrees

        Returns:
        --------
        colat_idx : int
            Co-latitude index
        """
        return 90 - lat

    def colat_to_lat(self, colat_idx):
        """
        Convert co-latitude index to latitude (degrees).

        Parameters:
        -----------
        colat_idx : int
            Co-latitude index

        Returns:
        --------
        lat : int
            Latitude in degrees
        """
        return 90 - colat_idx

    def _get_plot_timestamps(self, timestamps=None):
        """
        Get timestamps suitable for plotting.

        Parameters:
        -----------
        timestamps : list or None
            List of timestamps. If None, uses self.timestamps

        Returns:
        --------
        plot_times : list
            List of datetime objects or strings for plotting
        """
        if timestamps is None:
            timestamps = self.timestamps

        # Try to parse to datetime for better plotting
        parsed = []
        for ts in timestamps:
            try:
                parsed.append(self.parse_timestamp(ts, self.timestamp_format))
            except:
                parsed.append(ts)

        return parsed

    def _get_wrapped_slice(self, mlt_range):
        """
        Get slice for MLT range, handling wraparound.

        Parameters:
        -----------
        mlt_range : slice or tuple
            MLT range specification

        Returns:
        --------
        indices : list or slice
            Indices to extract
        """
        if isinstance(mlt_range, slice):
            return mlt_range

        start, stop = mlt_range
        if start < stop:
            return slice(start, stop)
        else:
            # Wraparound case
            return list(range(start, self.n_mlts)) + list(range(0, stop))

    @staticmethod
    @jit(nopython=True)
    def _rolling_corr_numba(x, y, window):
        """
        Compute rolling correlation using Numba for speed.

        Parameters:
        -----------
        x, y : ndarray
            1D arrays to correlate
        window : int
            Window size

        Returns:
        --------
        corrs : ndarray
            Rolling correlations
        """
        n = len(x)
        n_windows = n - window + 1
        corrs = np.empty(n_windows)

        for i in range(n_windows):
            x_win = x[i:i+window]
            y_win = y[i:i+window]

            x_mean = np.mean(x_win)
            y_mean = np.mean(y_win)

            x_centered = x_win - x_mean
            y_centered = y_win - y_mean

            numerator = np.sum(x_centered * y_centered)
            denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

            if denominator == 0:
                corrs[i] = 0
            else:
                corrs[i] = numerator / denominator

        return corrs

    @staticmethod
    @jit(nopython=True)
    def _autocorr_numba(x, nlags):
        """
        Compute autocorrelation function using Numba.

        Parameters:
        -----------
        x : ndarray
            1D array
        nlags : int
            Number of lags

        Returns:
        --------
        acf : ndarray
            Autocorrelation function
        """
        x_centered = x - np.mean(x)
        c0 = np.sum(x_centered**2)

        acf = np.empty(nlags + 1)
        acf[0] = 1.0

        for lag in range(1, nlags + 1):
            c_lag = np.sum(x_centered[:-lag] * x_centered[lag:])
            acf[lag] = c_lag / c0

        return acf

    # ==================== DETERMINISTIC ANALYSES ====================

    def plot_grid_cell_timeseries(self, lat_idx, mlt_idx):
        """
        Plot time series at a single grid cell.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees, e.g., 70 for 70°)
        mlt_idx : int
            MLT index
        """
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        plot_times = self._get_plot_timestamps()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_times, ampere, label='AMPERE', marker='o', markersize=4)
        ax.plot(plot_times, predicted, label='Predicted', marker='s', markersize=4)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=12)
        ax.set_title(f'Grid Cell Time Series (Lat={lat_idx}°, MLT={mlt_idx})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Calculate metrics
        correlation = np.corrcoef(ampere, predicted)[0, 1]
        rmse = np.sqrt(np.mean((ampere - predicted)**2))

        textstr = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

        return correlation, rmse

    def plot_regional_timeseries(self, mlt_idx=None, lat_range=None, mlt_range=None):
        """
        Plot regional time series.

        Two modes:
        1. Colormap mode (mlt_idx only): Shows all latitudes at single MLT
        2. Line plot mode (lat_range + mlt_range): Shows regional average

        Parameters:
        -----------
        mlt_idx : int or None
            Single MLT index for colormap mode
        lat_range : tuple or None
            (start_lat, end_lat) for line plot mode
        mlt_range : slice or None
            MLT range for line plot mode
        """
        if mlt_idx is not None and lat_range is None:
            # Colormap mode
            self._plot_regional_colormap(mlt_idx)
        elif lat_range is not None and mlt_range is not None:
            # Line plot mode
            self._plot_regional_lineplot(lat_range, mlt_range)
        else:
            raise ValueError("Specify either mlt_idx alone (colormap) or both lat_range and mlt_range (line plot)")

    def _plot_regional_colormap(self, mlt_idx):
        """Colormap mode for regional analysis."""
        ampere_data = self.ampere_stack[:, :, mlt_idx]
        predicted_data = self.predicted_stack[:, :, mlt_idx]
        residuals_data = ampere_data - predicted_data

        plot_times = self._get_plot_timestamps()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

        # Colormaps
        time_indices = np.arange(len(plot_times))
        lat_indices = np.arange(self.n_lats)

        vmax_data = max(np.abs(ampere_data).max(), np.abs(predicted_data).max())
        vmin_data = -vmax_data
        vmax_resid = np.abs(residuals_data).max()
        vmin_resid = -vmax_resid

        # AMPERE
        im1 = ax1.pcolormesh(time_indices, lat_indices, ampere_data.T,
                             cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data, shading='auto')
        ax1.invert_yaxis()
        plt.colorbar(im1, ax=ax1)
        ax1.set_ylabel('Latitude Index', fontsize=11)
        ax1.set_title(f'AMPERE (MLT={mlt_idx})', fontsize=12, fontweight='bold')

        # Predicted
        im2 = ax2.pcolormesh(time_indices, lat_indices, predicted_data.T,
                             cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data, shading='auto')
        ax2.invert_yaxis()
        plt.colorbar(im2, ax=ax2)
        ax2.set_ylabel('Latitude Index', fontsize=11)
        ax2.set_title(f'Predicted (MLT={mlt_idx})', fontsize=12, fontweight='bold')

        # Residuals
        im3 = ax3.pcolormesh(time_indices, lat_indices, residuals_data.T,
                             cmap='RdBu_r', vmin=vmin_resid, vmax=vmax_resid, shading='auto')
        ax3.invert_yaxis()
        plt.colorbar(im3, ax=ax3)
        ax3.set_ylabel('Latitude Index', fontsize=11)
        ax3.set_title('Residuals', fontsize=12, fontweight='bold')

        # Summary plots
        mean_ampere = np.mean(ampere_data, axis=0)
        mean_predicted = np.mean(predicted_data, axis=0)
        mean_residuals = np.mean(residuals_data, axis=0)

        ax4.plot(lat_indices, mean_ampere, 'b-', linewidth=2, label='AMPERE')
        ax4.plot(lat_indices, mean_predicted, 'r-', linewidth=2, label='Predicted')
        ax4.set_xlabel('Latitude Index', fontsize=11)
        ax4.set_ylabel(r'Mean Current Density $\mu$A/$m^2$', fontsize=11)
        ax4.set_title('Mean by Latitude', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5.plot(lat_indices, mean_residuals, 'g-', linewidth=2)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Latitude Index', fontsize=11)
        ax5.set_ylabel('Mean Residual', fontsize=11)
        ax5.set_title('Mean Residual by Latitude', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        std_residuals = np.std(residuals_data, axis=0)
        ax6.plot(lat_indices, std_residuals, 'orange', linewidth=2)
        ax6.set_xlabel('Latitude Index', fontsize=11)
        ax6.set_ylabel('Std Dev of Residuals', fontsize=11)
        ax6.set_title('Residual Std Dev by Latitude', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_regional_lineplot(self, lat_range, mlt_range):
        """Line plot mode for regional analysis."""
        colat_start = self.lat_to_colat(lat_range[1])
        colat_end = self.lat_to_colat(lat_range[0]) + 1

        ampere_region = self.ampere_stack[:, colat_start:colat_end, mlt_range]
        predicted_region = self.predicted_stack[:, colat_start:colat_end, mlt_range]

        ampere_mean = np.mean(ampere_region, axis=(1, 2))
        predicted_mean = np.mean(predicted_region, axis=(1, 2))
        residuals = ampere_mean - predicted_mean

        plot_times = self._get_plot_timestamps()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(plot_times, ampere_mean, 'b-', linewidth=2, label='AMPERE', marker='o')
        ax1.plot(plot_times, predicted_mean, 'r-', linewidth=2, label='Predicted', marker='s')
        ax1.set_ylabel(r'Regional Mean Current Density $\mu$A/$m^2$', fontsize=11)
        ax1.set_title(f'Regional Time Series (Lat {lat_range[0]}-{lat_range[1]}°, MLT range)',
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(plot_times, residuals, 'g-', linewidth=2, marker='o')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Residual', fontsize=11)
        ax2.set_title('Residuals (AMPERE - Predicted)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def cross_correlation_analysis(self, lat_idx, mlt_idx, max_lag=10):
        """
        Compute cross-correlation between AMPERE and predicted at different lags.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        max_lag : int
            Maximum lag to compute
        """
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        # Normalize
        ampere_norm = (ampere - ampere.mean()) / ampere.std()
        predicted_norm = (predicted - predicted.mean()) / predicted.std()

        # Compute cross-correlation using FFT (much faster)
        n = len(ampere_norm)
        cross_corr = signal.correlate(ampere_norm, predicted_norm, mode='same', method='fft') / n
        lags = signal.correlation_lags(n, n, mode='same')

        # Restrict to max_lag
        mask = np.abs(lags) <= max_lag

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lags[mask], cross_corr[mask], marker='o', linewidth=2)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero lag')
        ax.set_xlabel('Lag (time steps)', fontsize=12)
        ax.set_ylabel('Cross-correlation', fontsize=12)
        ax.set_title(f'Cross-Correlation Analysis (Lat={lat_idx}°, MLT={mlt_idx})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Find optimal lag
        max_corr_idx = np.argmax(cross_corr[mask])
        optimal_lag = lags[mask][max_corr_idx]
        max_corr = cross_corr[mask][max_corr_idx]

        textstr = f'Optimal lag: {optimal_lag} steps\nMax correlation: {max_corr:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

        return optimal_lag, max_corr

    def rolling_correlation_analysis(self, mlt_idx, window_size=10, lat_idx=None):
        """
        Compute rolling correlation.

        Two modes:
        1. Colormap (lat_idx=None): Show all latitudes
        2. Line plot (lat_idx specified): Single latitude

        Parameters:
        -----------
        mlt_idx : int
            MLT index
        window_size : int
            Rolling window size
        lat_idx : int or None
            Latitude for line plot mode
        """
        if lat_idx is None:
            # Colormap mode
            self._rolling_correlation_colormap(mlt_idx, window_size)
        else:
            # Line plot mode
            self._rolling_correlation_lineplot(lat_idx, mlt_idx, window_size)

    def _rolling_correlation_colormap(self, mlt_idx, window_size):
        """Colormap mode for rolling correlation."""
        n_windows = self.n_times - window_size + 1
        rolling_corrs_all = np.empty((self.n_lats, n_windows))

        for lat in range(self.n_lats):
            ampere = self.ampere_stack[:, lat, mlt_idx]
            predicted = self.predicted_stack[:, lat, mlt_idx]
            rolling_corrs_all[lat, :] = self._rolling_corr_numba(ampere, predicted, window_size)

        rolling_times = self.timestamps[window_size//2 : window_size//2 + n_windows]
        plot_times = self._get_plot_timestamps(rolling_times)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Colormap
        time_indices = np.arange(len(plot_times))
        lat_indices = np.arange(self.n_lats)

        im = ax1.pcolormesh(time_indices, lat_indices, rolling_corrs_all,
                            cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
        ax1.invert_yaxis()

        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Latitude Index', fontsize=11)
        ax1.set_title(f'Rolling Correlation (window={window_size}, MLT={mlt_idx})',
                      fontsize=13, fontweight='bold')

        # Summary
        mean_corr = rolling_corrs_all.mean(axis=1)
        std_corr = rolling_corrs_all.std(axis=1)

        ax2.plot(lat_indices, mean_corr, 'b-', linewidth=2, label='Mean')
        ax2.fill_between(lat_indices, mean_corr - std_corr, mean_corr + std_corr,
                         alpha=0.3, label='±1 std')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Latitude Index', fontsize=11)
        ax2.set_ylabel('Mean Correlation', fontsize=11)
        ax2.set_title('Average Rolling Correlation by Latitude', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _rolling_correlation_lineplot(self, lat_idx, mlt_idx, window_size):
        """Line plot mode for rolling correlation."""
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        rolling_corrs = self._rolling_corr_numba(ampere, predicted, window_size)

        rolling_times = self.timestamps[window_size//2 : window_size//2 + len(rolling_corrs)]
        plot_times = self._get_plot_timestamps(rolling_times)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_times, rolling_corrs, linewidth=2, marker='o')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title(f'Rolling Correlation (Lat={lat_idx}°, MLT={mlt_idx}, window={window_size})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def spectral_analysis(self, lat_idx, mlt_idx):
        """
        Frequency domain analysis using FFT.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        """
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        n = len(ampere)

        # Use rfft for real data (2x faster)
        fft_ampere = rfft(ampere)
        fft_pred = rfft(predicted)

        # Power spectral density
        psd_ampere = np.abs(fft_ampere)**2 / n
        psd_pred = np.abs(fft_pred)**2 / n

        freq = rfftfreq(n, d=1.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(freq[1:], psd_ampere[1:], label='AMPERE', linewidth=2)
        ax.semilogy(freq[1:], psd_pred[1:], label='Predicted', linewidth=2)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Power Spectral Density', fontsize=12)
        ax.set_title(f'Frequency Domain Analysis (Lat={lat_idx}°, MLT={mlt_idx})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def autocorrelation_analysis(self, lat_idx, mlt_idx, nlags=20):
        """
        Compute autocorrelation function.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        nlags : int
            Number of lags
        """
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        acf_ampere = self._autocorr_numba(ampere, nlags)
        acf_predicted = self._autocorr_numba(predicted, nlags)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.stem(range(nlags+1), acf_ampere)
        ax1.set_xlabel('Lag', fontsize=11)
        ax1.set_ylabel('Autocorrelation', fontsize=11)
        ax1.set_title(f'AMPERE Autocorrelation (Lat={lat_idx}°)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.stem(range(nlags+1), acf_predicted)
        ax2.set_xlabel('Lag', fontsize=11)
        ax2.set_ylabel('Autocorrelation', fontsize=11)
        ax2.set_title(f'Predicted Autocorrelation (Lat={lat_idx}°)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def hovmoller_diagram(self, fixed_mlt=None, fixed_lat=None):
        """
        Create Hovmöller diagram.

        Parameters:
        -----------
        fixed_mlt : int or None
            Fixed MLT for time vs latitude plot
        fixed_lat : int or None
            Fixed latitude for time vs MLT plot
        """
        if fixed_mlt is not None:
            # Time vs Latitude
            ampere_array = self.ampere_stack[:, :, fixed_mlt].T
            predicted_array = self.predicted_stack[:, :, fixed_mlt].T
            xlabel = 'Time Index'
            ylabel = 'Latitude Index'
            title_suffix = f'MLT={fixed_mlt}'
        elif fixed_lat is not None:
            # Time vs MLT
            colat_idx = self.lat_to_colat(fixed_lat)
            ampere_array = self.ampere_stack[:, colat_idx, :].T
            predicted_array = self.predicted_stack[:, colat_idx, :].T
            xlabel = 'Time Index'
            ylabel = 'MLT'
            title_suffix = f'Lat={fixed_lat}°'
        else:
            raise ValueError("Must specify either fixed_mlt or fixed_lat")

        diff = ampere_array - predicted_array

        vmax = max(np.abs(ampere_array).max(), np.abs(predicted_array).max())
        vmin = -vmax

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        im1 = ax1.imshow(ampere_array, aspect='auto', cmap='RdBu_r',
                         origin='upper', vmin=vmin, vmax=vmax)
        ax1.set_xlabel(xlabel, fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.set_title(f'AMPERE Hovmöller ({title_suffix})', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(predicted_array, aspect='auto', cmap='RdBu_r',
                         origin='upper', vmin=vmin, vmax=vmax)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(f'Predicted Hovmöller ({title_suffix})', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r', origin='upper')
        ax3.set_xlabel(xlabel, fontsize=11)
        ax3.set_ylabel(ylabel, fontsize=11)
        ax3.set_title('Difference (AMPERE - Predicted)', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.show()

    def dtw_analysis(self, lat_idx, mlt_idx, radius=10):
        """
        Dynamic Time Warping analysis (requires fastdtw package).

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        radius : int
            DTW search radius
        """
        try:
            from fastdtw import fastdtw
        except ImportError:
            print("fastdtw package not installed. Install with: pip install fastdtw")
            return

        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]

        distance, path = fastdtw(ampere, predicted, radius=radius)

        print(f"DTW Distance: {distance:.4f}")
        print(f"Path length: {len(path)}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        plot_times = self._get_plot_timestamps()
        ax1.plot(plot_times, ampere, label='AMPERE', linewidth=2)
        ax1.plot(plot_times, predicted, label='Predicted', linewidth=2)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
        ax1.set_title(f'Time Series (Lat={lat_idx}°, MLT={mlt_idx})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        path_array = np.array(path)
        ax2.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1)
        ax2.plot([0, len(ampere)], [0, len(predicted)], 'r--', linewidth=2, label='Perfect alignment')
        ax2.set_xlabel('AMPERE Index', fontsize=11)
        ax2.set_ylabel('Predicted Index', fontsize=11)
        ax2.set_title(f'DTW Alignment Path (Distance={distance:.2f})', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return distance, path

    # ==================== UNCERTAINTY-AWARE ANALYSES ====================

    def calibration_analysis(self, lat_idx=None, mlt_idx=None):
        """
        Analyze uncertainty calibration.

        Two modes:
        1. Single location (lat_idx and mlt_idx): Detailed diagnostics
        2. Spatial (no indices): Calibration maps

        Parameters:
        -----------
        lat_idx : int or None
            Latitude index
        mlt_idx : int or None
            MLT index
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        if lat_idx is not None and mlt_idx is not None:
            self._calibration_single_location(lat_idx, mlt_idx)
        else:
            self._calibration_spatial()

    def _calibration_single_location(self, lat_idx, mlt_idx):
        """Single location calibration analysis."""
        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]
        pred_std = self.std_stack[:, colat_idx, mlt_idx]

        residuals = ampere - predicted
        z_scores = residuals / pred_std

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Residuals vs uncertainty
        ax1 = axes[0, 0]
        ax1.scatter(pred_std, np.abs(residuals), alpha=0.6)
        ax1.plot([0, pred_std.max()], [0, pred_std.max()], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Predicted Std Dev', fontsize=11)
        ax1.set_ylabel('Absolute Residual', fontsize=11)
        ax1.set_title('Calibration: Uncertainty vs Error', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Z-score distribution
        ax2 = axes[0, 1]
        ax2.hist(z_scores, bins=30, density=True, alpha=0.7, label='Observed')
        x = np.linspace(-4, 4, 100)
        ax2.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
        ax2.set_xlabel('Z-score', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Calibration curve
        ax3 = axes[1, 0]
        confidence_levels = np.linspace(0, 3, 31)
        observed_coverage = []

        for conf in confidence_levels:
            within = np.abs(z_scores) <= conf
            observed_coverage.append(np.mean(within))

        expected_coverage = stats.norm.cdf(confidence_levels) - stats.norm.cdf(-confidence_levels)

        ax3.plot(expected_coverage, observed_coverage, 'b-', linewidth=2, label='Model')
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        ax3.set_xlabel('Expected Coverage', fontsize=11)
        ax3.set_ylabel('Observed Coverage', fontsize=11)
        ax3.set_title('Calibration Curve', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Q-Q plot
        ax4 = axes[1, 1]
        stats.probplot(z_scores, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Calculate metrics
        within_1sigma = np.mean(np.abs(z_scores) <= 1)
        within_2sigma = np.mean(np.abs(z_scores) <= 2)
        within_3sigma = np.mean(np.abs(z_scores) <= 3)

        print(f"\nCalibration Metrics (Lat={lat_idx}°, MLT={mlt_idx}):")
        print(f"  Coverage within 1σ: {within_1sigma:.1%} (expected: 68.3%)")
        print(f"  Coverage within 2σ: {within_2sigma:.1%} (expected: 95.4%)")
        print(f"  Coverage within 3σ: {within_3sigma:.1%} (expected: 99.7%)")
        print(f"  Z-score mean: {np.mean(z_scores):.3f} (ideal: 0)")
        print(f"  Z-score std: {np.std(z_scores):.3f} (ideal: 1)")

        plt.tight_layout()
        plt.show()

    def _calibration_spatial(self):
        """Spatial calibration analysis."""
        residuals = self.ampere_stack - self.predicted_stack
        z_scores = residuals / self.std_stack

        # Compute coverage for each location
        coverage_1sigma = np.mean(np.abs(z_scores) <= 1, axis=0)
        coverage_2sigma = np.mean(np.abs(z_scores) <= 2, axis=0)

        mean_uncertainty = np.mean(self.std_stack, axis=0)
        mean_error = np.mean(np.abs(residuals), axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1σ coverage
        im1 = axes[0, 0].imshow(coverage_1sigma, cmap='RdYlGn', vmin=0, vmax=1,
                                origin='upper', aspect='auto')
        axes[0, 0].set_title('Coverage within 1σ (target: 68.3%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('MLT')
        axes[0, 0].set_ylabel('Latitude Index')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2σ coverage
        im2 = axes[0, 1].imshow(coverage_2sigma, cmap='RdYlGn', vmin=0, vmax=1,
                                origin='upper', aspect='auto')
        axes[0, 1].set_title('Coverage within 2σ (target: 95.4%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('MLT')
        axes[0, 1].set_ylabel('Latitude Index')
        plt.colorbar(im2, ax=axes[0, 1])

        # Mean uncertainty
        im3 = axes[1, 0].imshow(mean_uncertainty, cmap='viridis', origin='upper', aspect='auto')
        axes[1, 0].set_title('Mean Predicted Uncertainty', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('MLT')
        axes[1, 0].set_ylabel('Latitude Index')
        plt.colorbar(im3, ax=axes[1, 0])

        # Mean error
        im4 = axes[1, 1].imshow(mean_error, cmap='plasma', origin='upper', aspect='auto')
        axes[1, 1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('MLT')
        axes[1, 1].set_ylabel('Latitude Index')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def confidence_interval_timeseries(self, lat_idx, mlt_idx, n_sigma=2):
        """
        Plot time series with confidence intervals.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        n_sigma : int
            Number of standard deviations for confidence bands
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]
        pred_std = self.std_stack[:, colat_idx, mlt_idx]

        plot_times = self._get_plot_timestamps()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Time series with confidence bands
        ax1.plot(plot_times, ampere, 'k-', linewidth=2, label='AMPERE', zorder=3)
        ax1.plot(plot_times, predicted, 'b-', linewidth=2, label='Predicted', zorder=2)

        for n in range(1, n_sigma + 1):
            alpha = 0.3 / n
            coverage = stats.norm.cdf(n) - stats.norm.cdf(-n)
            ax1.fill_between(plot_times,
                             predicted - n * pred_std,
                             predicted + n * pred_std,
                             alpha=alpha, color='blue',
                             label=f'±{n}σ ({coverage:.1%})')

        ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
        ax1.set_title(f'Time Series with Confidence Intervals (Lat={lat_idx}°, MLT={mlt_idx})',
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Predicted uncertainty vs actual error
        actual_error = np.abs(ampere - predicted)

        ax2.plot(plot_times, pred_std, 'r-', linewidth=2, label='Predicted Std')
        ax2.plot(plot_times, actual_error, 'g-', linewidth=2, label='Actual Error')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Magnitude', fontsize=11)
        ax2.set_title('Predicted Uncertainty vs Actual Error', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Calculate correlation between uncertainty and error
        corr = np.corrcoef(pred_std, actual_error)[0, 1]
        print(f"\nCorrelation between predicted uncertainty and actual error: {corr:.3f}")

    def crps_analysis(self, lat_idx=None, mlt_idx=None):
        """
        Continuous Ranked Probability Score analysis.

        Parameters:
        -----------
        lat_idx : int or None
            Latitude index
        mlt_idx : int or None
            MLT index
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        def crps_gaussian(observation, mean, std):
            """Compute CRPS for Gaussian distribution."""
            z = (observation - mean) / std
            crps = std * (z * (2 * stats.norm.cdf(z) - 1) +
                          2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
            return crps

        if lat_idx is not None and mlt_idx is not None:
            # Single location
            colat_idx = self.lat_to_colat(lat_idx)

            ampere = self.ampere_stack[:, colat_idx, mlt_idx]
            predicted = self.predicted_stack[:, colat_idx, mlt_idx]
            pred_std = self.std_stack[:, colat_idx, mlt_idx]

            crps_scores = np.array([crps_gaussian(obs, mu, sigma)
                                    for obs, mu, sigma in zip(ampere, predicted, pred_std)])

            plot_times = self._get_plot_timestamps()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(plot_times, crps_scores, linewidth=2)
            ax1.set_ylabel('CRPS', fontsize=11)
            ax1.set_title(f'CRPS Over Time (Lat={lat_idx}°, MLT={mlt_idx})',
                          fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Compare to MAE
            mae = np.abs(ampere - predicted)
            ax2.plot(plot_times, mae, label='MAE', linewidth=2)
            ax2.plot(plot_times, crps_scores, label='CRPS', linewidth=2)
            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('Score', fontsize=11)
            ax2.set_title('MAE vs CRPS Comparison', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print(f"\nCRPS Metrics:")
            print(f"  Mean CRPS: {np.mean(crps_scores):.4f}")
            print(f"  Mean MAE: {np.mean(mae):.4f}")
            print(f"  CRPS/MAE ratio: {np.mean(crps_scores)/np.mean(mae):.3f}")
        else:
            # Spatial map
            crps_map = np.zeros((self.n_lats, self.n_mlts))

            for i in range(self.n_lats):
                for j in range(self.n_mlts):
                    ampere = self.ampere_stack[:, i, j]
                    predicted = self.predicted_stack[:, i, j]
                    pred_std = self.std_stack[:, i, j]

                    crps_scores = np.array([crps_gaussian(obs, mu, sigma)
                                            for obs, mu, sigma in zip(ampere, predicted, pred_std)])
                    crps_map[i, j] = np.mean(crps_scores)

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(crps_map, cmap='plasma', origin='upper', aspect='auto')
            ax.set_title('Mean CRPS by Location', fontsize=14, fontweight='bold')
            ax.set_xlabel('MLT', fontsize=12)
            ax.set_ylabel('Latitude Index', fontsize=12)
            plt.colorbar(im, ax=ax, label='Mean CRPS')
            plt.tight_layout()
            plt.show()

    def uncertainty_error_relationship(self):
        """
        Analyze relationship between predicted uncertainty and actual errors.
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        residuals = self.ampere_stack - self.predicted_stack
        abs_errors = np.abs(residuals)

        errors_flat = abs_errors.flatten()
        uncertainty_flat = self.std_stack.flatten()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Scatter plot (hexbin for large datasets)
        ax1 = axes[0, 0]
        ax1.hexbin(uncertainty_flat, errors_flat, gridsize=50, cmap='viridis', mincnt=1)
        ax1.plot([0, uncertainty_flat.max()], [0, uncertainty_flat.max()],
                 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Predicted Uncertainty', fontsize=11)
        ax1.set_ylabel('Actual Error', fontsize=11)
        ax1.set_title('Uncertainty vs Error', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Binned analysis
        ax2 = axes[0, 1]
        n_bins = 20
        bins = np.linspace(0, np.percentile(uncertainty_flat, 95), n_bins)
        bin_indices = np.digitize(uncertainty_flat, bins)

        bin_means_uncertainty = []
        bin_means_error = []
        bin_std_error = []

        for i in range(1, n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 10:
                bin_means_uncertainty.append(np.mean(uncertainty_flat[mask]))
                bin_means_error.append(np.mean(errors_flat[mask]))
                bin_std_error.append(np.std(errors_flat[mask]))

        if len(bin_means_uncertainty) > 0:
            bin_means_uncertainty = np.array(bin_means_uncertainty)
            bin_means_error = np.array(bin_means_error)
            bin_std_error = np.array(bin_std_error)

            ax2.errorbar(bin_means_uncertainty, bin_means_error, yerr=bin_std_error,
                         fmt='o-', linewidth=2, capsize=5, label='Observed')
            ax2.plot([0, bin_means_uncertainty.max()], [0, bin_means_uncertainty.max()],
                     'r--', linewidth=2, label='Perfect')

        ax2.set_xlabel('Mean Predicted Uncertainty', fontsize=11)
        ax2.set_ylabel('Mean Actual Error', fontsize=11)
        ax2.set_title('Binned Analysis', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Spatial correlation map
        ax3 = axes[1, 0]

        corr_map = np.zeros((self.analyzer.n_lats, self.analyzer.n_mlts))
        for i in range(self.analyzer.n_lats):
            for j in range(self.analyzer.n_mlts):
                errors = abs_errors[:, i, j]
                uncert = self.analyzer.std_stack[:, i, j]
                if np.std(errors) > 0 and np.std(uncert) > 0:
                    corr_map[i, j] = np.corrcoef(errors, uncert)[0, 1]

        im3 = ax3.imshow(corr_map, cmap='RdBu_r', vmin=-1, vmax=1,
                        origin='upper', aspect='auto', rasterized=True)

        # Draw boxes around pixels with correlation > 0.5
        from matplotlib.patches import Rectangle

        for i in range(self.analyzer.n_lats):
            for j in range(self.analyzer.n_mlts):
                if corr_map[i, j] > 0.7:
                    # Rectangle centered on pixel: (x, y, width, height)
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False,
                                edgecolor='green',
                                linewidth=2)
                    ax3.add_patch(rect)
                elif corr_map[i, j] > 0.5:
                    # Rectangle centered on pixel: (x, y, width, height)
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False,
                                edgecolor='blue',
                                linewidth=2)
                    ax3.add_patch(rect)

        ax3.set_title('Correlation by Location', fontsize=11, fontweight='bold')
        ax3.set_xlabel('MLT', fontsize=10)
        ax3.set_ylabel('Latitude', fontsize=10)
        plt.colorbar(im3, ax=ax3, label='Correlation')

        # Plot 4: Ratio distribution
        ax4 = axes[1, 1]
        ratio = errors_flat / (uncertainty_flat + 1e-10)
        ax4.hist(ratio, bins=50, density=True, alpha=0.7, range=(0, 5))
        ax4.axvline(x=1, color='r', linestyle='--', linewidth=2, label='Ratio=1')
        ax4.set_xlabel('Error / Uncertainty', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Error-to-Uncertainty Ratio', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate metrics
        overall_corr = np.corrcoef(errors_flat, uncertainty_flat)[0, 1]
        median_ratio = np.median(ratio[ratio < 10])

        print(f"\nUncertainty-Error Relationship Metrics:")
        print(f"  Overall correlation: {overall_corr:.3f}")
        print(f"  Median error/uncertainty ratio: {median_ratio:.3f}")
        print(f"  Mean spatial correlation: {np.mean(corr_map):.3f}")

    def sharpness_analysis(self):
        """
        Analyze uncertainty sharpness (how confident the model is).
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        plot_times = self._get_plot_timestamps()

        mean_std_time = np.mean(self.std_stack, axis=(1, 2))
        median_std_time = np.median(self.std_stack, axis=(1, 2))
        mean_std_spatial = np.mean(self.std_stack, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Sharpness over time
        ax1 = axes[0, 0]
        ax1.plot(plot_times, mean_std_time, linewidth=2, label='Mean')
        ax1.plot(plot_times, median_std_time, linewidth=2, label='Median')
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Uncertainty (Std Dev)', fontsize=11)
        ax1.set_title('Sharpness Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Spatial distribution
        ax2 = axes[0, 1]
        im2 = ax2.imshow(mean_std_spatial, cmap='viridis', origin='upper', aspect='auto')
        ax2.set_title('Mean Uncertainty by Location', fontsize=12, fontweight='bold')
        ax2.set_xlabel('MLT', fontsize=11)
        ax2.set_ylabel('Latitude Index', fontsize=11)
        plt.colorbar(im2, ax=ax2, label='Std Dev')

        # Plot 3: Distribution
        ax3 = axes[1, 0]
        ax3.hist(self.std_stack.flatten(), bins=50, density=True, alpha=0.7)
        ax3.set_xlabel('Predicted Std Dev', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Distribution of Uncertainties', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Coefficient of variation
        ax4 = axes[1, 1]
        cv_time = np.std(self.std_stack, axis=(1, 2)) / (mean_std_time + 1e-10)
        ax4.plot(plot_times, cv_time, linewidth=2)
        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Coefficient of Variation', fontsize=11)
        ax4.set_title('Spatial Variability of Uncertainty', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

        print(f"\nSharpness Metrics:")
        print(f"  Overall mean uncertainty: {np.mean(self.std_stack):.4f}")
        print(f"  Overall median uncertainty: {np.median(self.std_stack):.4f}")
        print(f"  Temporal variability (CV): {np.std(mean_std_time)/np.mean(mean_std_time):.3f}")
        print(f"  Spatial variability (CV): {np.std(mean_std_spatial)/np.mean(mean_std_spatial):.3f}")

    def probabilistic_skill_scores(self, threshold=None):
        """
        Compute probabilistic skill scores for threshold exceedance.

        Parameters:
        -----------
        threshold : float or None
            Threshold value. If None, uses 90th percentile.
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        if threshold is None:
            threshold = np.percentile(self.ampere_stack, 90)

        # Compute probability of exceeding threshold
        z_threshold = (threshold - self.predicted_stack) / self.std_stack
        prob_exceed = 1 - stats.norm.cdf(z_threshold)

        # Observed exceedance
        observed_exceed = (self.ampere_stack > threshold).astype(float)

        # Brier score
        brier_score = np.mean((prob_exceed - observed_exceed)**2)

        # Brier skill score (relative to climatology)
        climatology_prob = np.mean(observed_exceed)
        brier_score_clim = np.mean((climatology_prob - observed_exceed)**2)
        brier_skill_score = 1 - brier_score / brier_score_clim

        # Reliability diagram
        n_bins = 10
        prob_bins = np.linspace(0, 1, n_bins + 1)

        observed_freq = []
        forecast_prob = []
        counts = []

        for i in range(n_bins):
            mask = (prob_exceed >= prob_bins[i]) & (prob_exceed < prob_bins[i+1])
            if np.sum(mask) > 0:
                observed_freq.append(np.mean(observed_exceed[mask]))
                forecast_prob.append(np.mean(prob_exceed[mask]))
                counts.append(np.sum(mask))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Plot 1: Reliability diagram
        ax1 = axes[0]
        if len(forecast_prob) > 0:
            sizes = np.array(counts) / np.sum(counts) * 1000
            ax1.scatter(forecast_prob, observed_freq, s=sizes, alpha=0.6)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Forecast Probability', fontsize=11)
        ax1.set_ylabel('Observed Frequency', fontsize=11)
        ax1.set_title(f'Reliability Diagram (Threshold={threshold:.2f})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Plot 2: Probability vs actual values
        ax2 = axes
        # Plot 2: Probability vs actual values
        ax2 = axes[1]
        sample_indices = np.random.choice(prob_exceed.size, min(5000, prob_exceed.size), replace=False)
        ax2.scatter(prob_exceed.flatten()[sample_indices],
                   self.ampere_stack.flatten()[sample_indices],
                   alpha=0.3, s=1)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Forecast Probability', fontsize=11)
        ax2.set_ylabel(r'Actual Current Density $\mu$A/$m^2$', fontsize=11)
        ax2.set_title('Probability vs Observations', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Forecast distribution
        ax3 = axes[2]
        ax3.hist(prob_exceed.flatten(), bins=30, density=True, alpha=0.7)
        ax3.axvline(x=climatology_prob, color='r', linestyle='--',
                   linewidth=2, label=f'Climatology ({climatology_prob:.2%})')
        ax3.set_xlabel('Forecast Probability', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Forecast Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nProbabilistic Skill Scores (Threshold={threshold:.3f}):")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Brier Skill Score: {brier_skill_score:.4f}")
        print(f"  Climatology probability: {climatology_prob:.2%}")

    def pseudo_ensemble_analysis(self, lat_idx, mlt_idx, n_samples=50):
        """
        Generate pseudo-ensemble from mean and std and analyze.

        Parameters:
        -----------
        lat_idx : int
            Latitude index (in degrees)
        mlt_idx : int
            MLT index
        n_samples : int
            Number of ensemble members to generate
        """
        if not self.has_uncertainty:
            print("Uncertainty data not available")
            return

        colat_idx = self.lat_to_colat(lat_idx)

        ampere = self.ampere_stack[:, colat_idx, mlt_idx]
        predicted = self.predicted_stack[:, colat_idx, mlt_idx]
        pred_std = self.std_stack[:, colat_idx, mlt_idx]

        n_times = len(ampere)

        # Generate ensemble
        ensemble = np.random.normal(
            loc=predicted[:, np.newaxis],
            scale=pred_std[:, np.newaxis],
            size=(n_times, n_samples)
        )

        plot_times = self._get_plot_timestamps()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Spaghetti plot
        ax1 = axes[0]
        for i in range(min(20, n_samples)):
            ax1.plot(plot_times, ensemble[:, i], 'b-', alpha=0.2, linewidth=0.5)
        ax1.plot(plot_times, predicted, 'b-', linewidth=2, label='Ensemble mean')
        ax1.plot(plot_times, ampere, 'k-', linewidth=2, label='Observed')
        ax1.fill_between(plot_times,
                         np.percentile(ensemble, 25, axis=1),
                         np.percentile(ensemble, 75, axis=1),
                         alpha=0.3, color='blue', label='IQR')
        ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
        ax1.set_title(f'Pseudo-Ensemble Forecast (Lat={lat_idx}°, MLT={mlt_idx})',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Rank histogram
        ax2 = axes[1]
        ranks = np.zeros(n_times, dtype=int)
        for t in range(n_times):
            ranks[t] = np.sum(ensemble[t, :] < ampere[t])

        ax2.hist(ranks, bins=np.arange(n_samples + 2) - 0.5, density=True, alpha=0.7)
        ax2.axhline(y=1/(n_samples+1), color='r', linestyle='--',
                   linewidth=2, label='Uniform (perfect)')
        ax2.set_xlabel('Rank of Observation', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Rank Histogram (Should be Uniform)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Spread-skill relationship
        ax3 = axes[2]
        ensemble_spread = np.std(ensemble, axis=1)
        errors = np.abs(ampere - predicted)

        ax3.scatter(ensemble_spread, errors, alpha=0.6)
        max_val = max(ensemble_spread.max(), errors.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect spread-skill')
        ax3.set_xlabel('Ensemble Spread (Std Dev)', fontsize=11)
        ax3.set_ylabel('Forecast Error (Absolute)', fontsize=11)
        ax3.set_title('Spread-Skill Relationship', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Calculate metrics
        spread_skill_ratio = np.mean(ensemble_spread) / np.mean(errors)
        rank_uniformity = stats.chisquare(np.bincount(ranks, minlength=n_samples+1))[1]

        print(f"\nPseudo-Ensemble Metrics:")
        print(f"  Spread-skill ratio: {spread_skill_ratio:.3f} (ideal: 1.0)")
        print(f"  Rank histogram uniformity p-value: {rank_uniformity:.4f}")


class AnalysisReportGenerator:
    """
    Generates comprehensive PDF reports from model analysis results.
    Includes ALL analyses from PostProcessingAnalysis class.
    """

    def __init__(self, analyzer, output_filename=None, lat_idx=None, mlt_idx=None,
                 lat_range=None, mlt_range=None, plot_time_range=None,
                 dpi=100, compress_images=True):
        """
        Initialize report generator.

        Parameters:
        -----------
        analyzer : PostProcessingAnalysis
            Initialized analysis object
        output_filename : str or None
            Output PDF filename. If None, auto-generates with timestamp
        lat_idx : int or None
            Single latitude index for analyses. Mutually exclusive with lat_range.
        mlt_idx : int or None
            Single MLT index for analyses. Mutually exclusive with mlt_range.
        lat_range : tuple or None
            (start_lat, end_lat) for averaging. Mutually exclusive with lat_idx.
            Example: (65, 75) averages latitudes from 65° to 75°
        mlt_range : tuple or None
            (start_mlt, end_mlt) for averaging. Mutually exclusive with mlt_idx.
            Example: (9, 12) averages MLT bins from 9 to 12
            Can wrap around: (22, 2) averages MLTs 22, 23, 0, 1, 2
        plot_time_range : tuple or None
            (start_time, end_time) for grid cell and regional plots.
        dpi : int
            Resolution for figures (default: 100).
        compress_images : bool
            If True, uses image compression (default: True)
        """
        self.analyzer = analyzer

        # Validate mutually exclusive parameters
        if lat_idx is not None and lat_range is not None:
            raise ValueError("Cannot specify both lat_idx and lat_range. Choose one.")
        if mlt_idx is not None and mlt_range is not None:
            raise ValueError("Cannot specify both mlt_idx and mlt_range. Choose one.")

        # Set up latitude handling
        if lat_range is not None:
            self.lat_idx = None
            self.lat_range = lat_range
            self.lat_mode = 'range'
            # Convert to co-latitude indices
            colat_start = self.analyzer.lat_to_colat(lat_range[1])  # Higher lat = lower colat
            colat_end = self.analyzer.lat_to_colat(lat_range[0]) + 1  # +1 for inclusive
            self.colat_slice = slice(colat_start, colat_end)
        elif lat_idx is not None:
            self.lat_idx = lat_idx
            self.lat_range = None
            self.lat_mode = 'single'
            self.colat_slice = self.analyzer.lat_to_colat(lat_idx)
        else:
            raise ValueError("Must specify either lat_idx or lat_range")

        # Set up MLT handling
        if mlt_range is not None:
            self.mlt_idx = None
            self.mlt_range = mlt_range
            self.mlt_mode = 'range'
            # Note: Don't create slice here if it wraps around
            # The slice will be handled in _extract_data
            if mlt_range[0] <= mlt_range[1]:
                # Normal range
                self.mlt_slice = slice(mlt_range[0], mlt_range[1])
            else:
                # Wraparound - store slice as None, handle in _extract_data
                self.mlt_slice = None
        elif mlt_idx is not None:
            self.mlt_idx = mlt_idx
            self.mlt_range = None
            self.mlt_mode = 'single'
            self.mlt_slice = mlt_idx
        else:
            raise ValueError("Must specify either mlt_idx or mlt_range")

        self.plot_time_range = plot_time_range
        self.dpi = dpi
        self.compress_images = compress_images

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"model_analysis_report_{timestamp}.pdf"

        self.output_filename = output_filename

        # Storage for metrics
        self.metrics = {}
        self.executive_summary = []

    def add_summary_statement(self, statement, category="General"):
        """Add a statement to the executive summary."""
        self.executive_summary.append({
            'category': category,
            'statement': statement
        })

    def _save_figure(self, pdf, fig):
        """
        Save figure to PDF with compression settings.
        """
        if self.compress_images:
            # Save with compression
            pdf.savefig(fig, bbox_inches='tight', dpi=self.dpi)
        else:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _get_location_string(self):
        """Generate a string describing the analysis location."""
        if self.lat_mode == 'range':
            lat_str = f"Lat {self.lat_range[0]}-{self.lat_range[1]}°"
        else:
            lat_str = f"Lat={self.lat_idx}°"

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                # Wraparound case
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
            else:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
        else:
            mlt_str = f"MLT={self.mlt_idx}"

        return f"{lat_str}, {mlt_str}"

    def _extract_data(self, data_array):
        """
        Extract and potentially average data based on lat/mlt mode.
        Handles MLT wraparound (e.g., mlt_range=(22, 2) wraps around midnight).

        Parameters:
        -----------
        data_array : ndarray
            3D array with shape (time, lat, mlt) or 1D array with shape (time,)

        Returns:
        --------
        extracted_data : ndarray
            1D array with shape (time,) after extraction and averaging
        """
        if data_array.ndim == 1:
            return data_array  # Already 1D

        # Check if MLT range wraps around (e.g., 22 to 2 means 22,23,0,1,2)
        mlt_wraps = False
        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                mlt_wraps = True

        # Handle MLT extraction with potential wraparound
        if mlt_wraps:
            # Extract two segments and concatenate
            # Segment 1: from start to end of array (e.g., 22, 23)
            # Segment 2: from beginning to end point (e.g., 0, 1, 2)
            segment1 = data_array[:, :, self.mlt_range[0]:]
            segment2 = data_array[:, :, :self.mlt_range[1]]
            mlt_subset = np.concatenate([segment1, segment2], axis=2)
        else:
            # Normal extraction (no wraparound)
            if self.mlt_mode == 'range':
                mlt_subset = data_array[:, :, self.mlt_slice]
            else:
                mlt_subset = data_array

        # Now handle latitude extraction and averaging
        if self.lat_mode == 'range' and self.mlt_mode == 'range':
            # Both ranges: extract lat subset and average over both dimensions
            if mlt_wraps:
                lat_mlt_subset = mlt_subset[:, self.colat_slice, :]
            else:
                lat_mlt_subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(lat_mlt_subset, axis=(1, 2))

        elif self.lat_mode == 'range' and self.mlt_mode == 'single':
            # Lat range, single MLT: extract MLT column and average over latitudes
            subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(subset, axis=1)

        elif self.lat_mode == 'single' and self.mlt_mode == 'range':
            # Single lat, MLT range: extract lat row and average over MLTs
            if mlt_wraps:
                subset = mlt_subset[:, self.colat_slice, :]
            else:
                subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(subset, axis=1)

        else:
            # Both single: extract single point
            return data_array[:, self.colat_slice, self.mlt_slice]

    def _filter_to_plot_time_range(self, timestamps, *data_arrays):
        """
        Filter timestamps and data arrays to plot time range.

        Returns:
        --------
        filtered_timestamps, *filtered_data_arrays
        """
        if self.plot_time_range is None:
            return (timestamps,) + data_arrays

        start_time, end_time = self.plot_time_range

        # Parse if strings
        if isinstance(start_time, str):
            start_time = self.analyzer.parse_timestamp(start_time, self.analyzer.timestamp_format)
        if isinstance(end_time, str):
            end_time = self.analyzer.parse_timestamp(end_time, self.analyzer.timestamp_format)

        # Parse timestamps if needed
        if isinstance(timestamps[0], str):
            parsed_timestamps = [self.analyzer.parse_timestamp(t, self.analyzer.timestamp_format)
                                for t in timestamps]
        else:
            parsed_timestamps = timestamps

        # Find indices
        mask = [(start_time <= t <= end_time) for t in parsed_timestamps]
        indices = np.where(mask)[0]

        if len(indices) == 0:
            # No data in range, return all
            return (timestamps,) + data_arrays

        filtered_timestamps = [timestamps[i] for i in indices]
        filtered_arrays = tuple(arr[indices] if arr.ndim == 1 else arr[indices, ...]
                               for arr in data_arrays)

        return (filtered_timestamps,) + filtered_arrays

    def create_title_page(self, pdf):
        """Create title page with report metadata."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.85, 'Model Analysis Report',
                ha='center', va='top', fontsize=24, fontweight='bold')

        # Date
        date_str = datetime.now().strftime("%B %d, %Y")
        ax.text(0.5, 0.78, f'Generated: {date_str}',
                ha='center', va='top', fontsize=12)

        # Dataset info
        info_y = 0.65
        ax.text(0.5, info_y, 'Dataset Information',
                ha='center', va='top', fontsize=16, fontweight='bold')

        info_text = [
            f"Number of timesteps: {self.analyzer.n_times}",
            f"Spatial grid: {self.analyzer.n_lats} latitudes × {self.analyzer.n_mlts} MLTs",
            f"Time range: {self.analyzer.timestamps[0]} to {self.analyzer.timestamps[-1]}",
            f"Uncertainty available: {'Yes' if self.analyzer.has_uncertainty else 'No'}",
            f"Analysis location: {self._get_location_string()}"
        ]

        if self.plot_time_range is not None:
            info_text.append(f"Plot time range: {self.plot_time_range[0]} to {self.plot_time_range[1]}")

        for i, text in enumerate(info_text):
            ax.text(0.5, info_y - 0.07 - i*0.04, text,
                    ha='center', va='top', fontsize=11)

        # Footer
        ax.text(0.5, 0.1, 'This report contains comprehensive validation metrics\nfor geomagnetic model predictions',
                ha='center', va='top', fontsize=10, style='italic')

        self._save_figure(pdf, fig)

    def create_executive_summary_page(self, pdf):
        """Create executive summary page with key findings."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.98, 'Executive Summary',
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax.transAxes)

        # Group statements by category
        categories = {}
        for item in self.executive_summary:
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item['statement'])

        # Write summary
        y_pos = 0.92
        line_height = 0.025

        for category, statements in categories.items():
            # Category header
            ax.text(0.05, y_pos, f"{category}:",
                    ha='left', va='top', fontsize=12, fontweight='bold',
                    transform=ax.transAxes)
            y_pos -= line_height * 1.2

            # Statements (without bullet points)
            for statement in statements:
                # Wrap long statements
                wrapped = self._wrap_text(statement, 95)
                for line in wrapped:
                    ax.text(0.05, y_pos, line,
                            ha='left', va='top', fontsize=10,
                            transform=ax.transAxes)
                    y_pos -= line_height

                    if y_pos < 0.05:  # Start new page if needed
                        self._save_figure(pdf, fig)
                        fig = plt.figure(figsize=(8.5, 11))
                        ax = fig.add_subplot(111)
                        ax.axis('off')
                        y_pos = 0.98

            y_pos -= line_height * 0.5  # Space between categories

        self._save_figure(pdf, fig)

    @staticmethod
    def _wrap_text(text, width):
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def run_grid_cell_analysis(self, pdf):
        """Run and document grid cell time series analysis."""
        print("Running grid cell analysis...")

        # Extract data using new method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)

        # Apply time filtering for plotting
        plot_timestamps = self.analyzer.timestamps
        plot_ampere = ampere
        plot_predicted = predicted

        if self.analyzer.has_uncertainty:
            pred_std = self._extract_data(self.analyzer.std_stack)
            plot_timestamps, plot_ampere, plot_predicted, plot_std = self._filter_to_plot_time_range(
                plot_timestamps, plot_ampere, plot_predicted, pred_std
            )
        else:
            plot_timestamps, plot_ampere, plot_predicted = self._filter_to_plot_time_range(
                plot_timestamps, plot_ampere, plot_predicted
            )

        plot_times = self.analyzer._get_plot_timestamps(plot_timestamps)

        if self.analyzer.has_uncertainty:
            # Show with uncertainty bands
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

            # Plot 1: Time series with confidence bands
            from scipy import stats
            ax1.plot(plot_times, plot_ampere, 'k-', linewidth=2, label='AMPERE (actual)', zorder=3)
            ax1.plot(plot_times, plot_predicted, 'b-', linewidth=2, label='Predicted mean', zorder=2)

            n_sigma = 2
            for n in range(1, n_sigma + 1):
                alpha = 0.3 / n
                coverage = stats.norm.cdf(n) - stats.norm.cdf(-n)
                ax1.fill_between(plot_times,
                                 plot_predicted - n * plot_std,
                                 plot_predicted + n * plot_std,
                                 alpha=alpha, color='blue',
                                 label=f'±{n}σ ({coverage:.1%})')

            ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
            title_str = f'Time Series with Uncertainty ({self._get_location_string()})'
            if self.plot_time_range is not None:
                title_str += f'\n[{self.plot_time_range[0]} to {self.plot_time_range[1]}]'
            ax1.set_title(title_str, fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Uncertainty evolution
            ax2.plot(plot_times, plot_std, 'r-', linewidth=2, label='Predicted uncertainty')
            ax2_twin = ax2.twinx()
            residuals = np.abs(plot_ampere - plot_predicted)
            ax2_twin.plot(plot_times, residuals, 'g-', linewidth=2, alpha=0.7, label='Actual error')

            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('Predicted Std Dev', color='r', fontsize=11)
            ax2_twin.set_ylabel('Actual Absolute Error', color='g', fontsize=11)
            ax2.set_title('Uncertainty Evolution', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2_twin.tick_params(axis='y', labelcolor='g')
            ax2.grid(True, alpha=0.3)

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Calculate metrics on FULL dataset (not just plot range)
            uncertainty_error_corr = np.corrcoef(pred_std, np.abs(ampere - predicted))[0, 1]

        else:
            fig, ax1 = plt.subplots(figsize=(11, 6))
            ax1.plot(plot_times, plot_ampere, label='AMPERE', marker='o', markersize=3)
            ax1.plot(plot_times, plot_predicted, label='Predicted', marker='s', markersize=3)
            ax1.set_xlabel('Time', fontsize=11)
            ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
            title_str = f'Time Series ({self._get_location_string()})'
            if self.plot_time_range is not None:
                title_str += f'\n[{self.plot_time_range[0]} to {self.plot_time_range[1]}]'
            ax1.set_title(title_str, fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

        # Calculate metrics on FULL dataset
        correlation = np.corrcoef(ampere, predicted)[0, 1]
        rmse = np.sqrt(np.mean((ampere - predicted)**2))
        mae = np.mean(np.abs(ampere - predicted))
        bias = np.mean(predicted - ampere)

        self.metrics['grid_cell'] = {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'bias': bias
        }

        if self.analyzer.has_uncertainty:
            self.metrics['grid_cell']['uncertainty_error_corr'] = uncertainty_error_corr

        textstr_lines = [
            f'Correlation: {correlation:.3f}',
            f'RMSE: {rmse:.3f}',
            f'MAE: {mae:.3f}',
            f'Bias: {bias:.3f}'
        ]

        if self.analyzer.has_uncertainty:
            textstr_lines.append(f'Unc-Err Corr: {uncertainty_error_corr:.3f}')

        textstr = '\n'.join(textstr_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        self._save_figure(pdf, fig)

        if correlation > 0.9:
            quality = "excellent"
        elif correlation > 0.7:
            quality = "good"
        elif correlation > 0.5:
            quality = "moderate"
        else:
            quality = "poor"

        base_statement = (
            f"Point prediction at {self._get_location_string()} shows {quality} correlation "
            f"(r={correlation:.3f}) with RMSE={rmse:.3f}. "
            f"Model exhibits {'positive' if bias > 0 else 'negative'} bias of {abs(bias):.3f}."
        )

        if self.analyzer.has_uncertainty:
            if uncertainty_error_corr > 0.5:
                unc_quality = "strong"
            elif uncertainty_error_corr > 0.3:
                unc_quality = "moderate"
            else:
                unc_quality = "weak"

            base_statement += (
                f" Uncertainty-error correlation is {uncertainty_error_corr:.3f}, indicating "
                f"{unc_quality} awareness of prediction quality."
            )

        self.add_summary_statement(base_statement, category="Point Performance")

    def run_regional_analysis(self, pdf):
        """Run and document regional time series analysis."""
        print("Running regional analysis...")

        # For regional analysis, we show the 2D map at the specified MLT
        # If mlt_mode is 'range', we average over the MLT range first
        if self.mlt_mode == 'range':
            # Check for wraparound
            if self.mlt_range[0] > self.mlt_range[1]:
                # Wraparound: concatenate two segments
                segment1 = self.analyzer.ampere_stack[:, :, self.mlt_range[0]:]
                segment2 = self.analyzer.ampere_stack[:, :, :self.mlt_range[1]]
                ampere_data = np.mean(np.concatenate([segment1, segment2], axis=2), axis=2)

                segment1 = self.analyzer.predicted_stack[:, :, self.mlt_range[0]:]
                segment2 = self.analyzer.predicted_stack[:, :, :self.mlt_range[1]]
                predicted_data = np.mean(np.concatenate([segment1, segment2], axis=2), axis=2)
            else:
                # Normal range
                ampere_data = np.mean(self.analyzer.ampere_stack[:, :, self.mlt_slice], axis=2)
                predicted_data = np.mean(self.analyzer.predicted_stack[:, :, self.mlt_slice], axis=2)
        else:
            # Single MLT
            ampere_data = self.analyzer.ampere_stack[:, :, self.mlt_slice]
            predicted_data = self.analyzer.predicted_stack[:, :, self.mlt_slice]

        residuals_data = ampere_data - predicted_data

        # Apply time filtering for plotting
        plot_timestamps = self.analyzer.timestamps
        plot_ampere, plot_predicted, plot_residuals = ampere_data, predicted_data, residuals_data

        plot_timestamps, plot_ampere, plot_predicted, plot_residuals = self._filter_to_plot_time_range(
            plot_timestamps, plot_ampere, plot_predicted, plot_residuals
        )

        plot_times = self.analyzer._get_plot_timestamps(plot_timestamps)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 13))

        time_indices = np.arange(len(plot_times))
        lat_indices = np.arange(self.analyzer.n_lats)

        vmax_data = max(np.abs(plot_ampere).max(), np.abs(plot_predicted).max())
        vmin_data = -vmax_data
        vmax_resid = np.abs(plot_residuals).max()
        vmin_resid = -vmax_resid

        # AMPERE
        im1 = ax1.pcolormesh(time_indices, lat_indices, plot_ampere.T,
                            cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data, shading='auto',
                            rasterized=True)
        ax1.invert_yaxis()
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label(r'Current Density $\mu$A/$m^2$', rotation=270, labelpad=15)
        ax1.set_ylabel('Latitude', fontsize=11)

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                title_str = f'AMPERE Data (MLT {self.mlt_range[0]}-{self.mlt_range[1]} avg)'
            else:
                title_str = f'AMPERE Data (MLT {self.mlt_range[0]}-{self.mlt_range[1]} avg)'
        else:
            title_str = f'AMPERE Data (MLT={self.mlt_idx})'

        if self.plot_time_range is not None:
            title_str += f'\n[{self.plot_time_range[0]} to {self.plot_time_range[1]}]'
        ax1.set_title(title_str, fontsize=12, fontweight='bold')

        lat_tick_indices = np.arange(0, self.analyzer.n_lats, max(1, self.analyzer.n_lats//10))
        ax1.set_yticks(lat_tick_indices)
        ax1.set_yticklabels([f'{self.analyzer.colat_to_lat(i)}°' for i in lat_tick_indices])

        # Predicted
        im2 = ax2.pcolormesh(time_indices, lat_indices, plot_predicted.T,
                            cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data, shading='auto',
                            rasterized=True)
        ax2.invert_yaxis()
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(r'Current Density $\mu$A/$m^2$', rotation=270, labelpad=15)
        ax2.set_ylabel('Latitude', fontsize=11)

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                ax2.set_title(f'Predicted Data (MLT {self.mlt_range[0]}-{self.mlt_range[1]} avg)',
                             fontsize=12, fontweight='bold')
            else:
                ax2.set_title(f'Predicted Data (MLT {self.mlt_range[0]}-{self.mlt_range[1]} avg)',
                             fontsize=12, fontweight='bold')
        else:
            ax2.set_title(f'Predicted Data (MLT={self.mlt_idx})', fontsize=12, fontweight='bold')

        ax2.set_yticks(lat_tick_indices)
        ax2.set_yticklabels([f'{self.analyzer.colat_to_lat(i)}°' for i in lat_tick_indices])

        # Residuals
        im3 = ax3.pcolormesh(time_indices, lat_indices, plot_residuals.T,
                            cmap='RdBu_r', vmin=vmin_resid, vmax=vmax_resid, shading='auto',
                            rasterized=True)
        ax3.invert_yaxis()
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Residual', rotation=270, labelpad=15)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Latitude', fontsize=11)
        ax3.set_title('Residuals (AMPERE - Predicted)', fontsize=12, fontweight='bold')
        ax3.set_yticks(lat_tick_indices)
        ax3.set_yticklabels([f'{self.analyzer.colat_to_lat(i)}°' for i in lat_tick_indices])

        n_ticks = min(8, len(plot_times))
        tick_indices = np.linspace(0, len(plot_times)-1, n_ticks, dtype=int)

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks(tick_indices)
            if ax == ax3:
                ax.set_xticklabels([plot_times[i].strftime('%H:%M') if hasattr(plot_times[i], 'strftime')
                                   else str(plot_times[i]) for i in tick_indices], rotation=45)
            else:
                ax.set_xticklabels([])

        plt.tight_layout()
        self._save_figure(pdf, fig)

        # Calculate metrics on FULL dataset
        spatial_rmse = np.sqrt(np.mean(residuals_data**2, axis=0))
        mean_spatial_rmse = np.mean(spatial_rmse)
        max_spatial_rmse = np.max(spatial_rmse)

        self.metrics['regional'] = {
            'mean_spatial_rmse': mean_spatial_rmse,
            'max_spatial_rmse': max_spatial_rmse
        }

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
            else:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
        else:
            mlt_str = f"MLT={self.mlt_idx}"

        self.add_summary_statement(
            f"Spatial analysis at {mlt_str} shows mean RMSE of {mean_spatial_rmse:.3f} "
            f"across all latitudes. Maximum errors ({max_spatial_rmse:.3f}) occur at "
            f"{'high' if np.argmax(spatial_rmse) < self.analyzer.n_lats//2 else 'low'} latitudes.",
            category="Spatial Performance"
        )

    def run_cross_correlation_analysis(self, pdf):
        """Run and document cross-correlation analysis."""
        print("Running cross-correlation analysis...")

        # Extract data using new method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)

        ampere_norm = (ampere - ampere.mean()) / ampere.std()
        predicted_norm = (predicted - predicted.mean()) / predicted.std()

        from scipy import signal
        n = len(ampere_norm)
        cross_corr = signal.correlate(ampere_norm, predicted_norm, mode='same', method='fft') / n
        lags = signal.correlation_lags(n, n, mode='same')

        max_lag = 10
        mask = np.abs(lags) <= max_lag

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lags[mask], cross_corr[mask], marker='o', linewidth=2)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero lag')
        ax.set_xlabel('Lag (time steps)', fontsize=11)
        ax.set_ylabel('Cross-correlation', fontsize=11)
        ax.set_title(f'Cross-Correlation Analysis ({self._get_location_string()})',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        max_corr_idx = np.argmax(cross_corr[mask])
        optimal_lag = lags[mask][max_corr_idx]
        max_corr = cross_corr[mask][max_corr_idx]

        self.metrics['cross_correlation'] = {
            'optimal_lag': optimal_lag,
            'max_correlation': max_corr
        }

        textstr = f'Optimal lag: {optimal_lag} steps\nMax correlation: {max_corr:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        self._save_figure(pdf, fig)

        if optimal_lag == 0:
            timing_msg = "Model predictions are temporally well-aligned with observations"
        elif optimal_lag > 0:
            timing_msg = f"Model lags observations by {optimal_lag} timesteps (reacts too slowly)"
        else:
            timing_msg = f"Model leads observations by {abs(optimal_lag)} timesteps (anticipates changes)"

        self.add_summary_statement(
            f"{timing_msg}. Maximum achievable correlation is {max_corr:.3f}.",
            category="Temporal Alignment"
        )

    def run_rolling_correlation_analysis(self, pdf):
        """Run and document rolling correlation analysis."""
        print("Running rolling correlation analysis...")

        window_size = 10
        n_windows = self.analyzer.n_times - window_size + 1
        rolling_corrs_all = np.empty((self.analyzer.n_lats, n_windows))

        # For rolling correlation, we still compute for all latitudes
        # but we average over MLT range if specified
        for lat in range(self.analyzer.n_lats):
            if self.mlt_mode == 'range':
                # Check for wraparound
                if self.mlt_range[0] > self.mlt_range[1]:
                    # Wraparound
                    segment1 = self.analyzer.ampere_stack[:, lat, self.mlt_range[0]:]
                    segment2 = self.analyzer.ampere_stack[:, lat, :self.mlt_range[1]]
                    ampere = np.mean(np.concatenate([segment1, segment2], axis=1), axis=1)

                    segment1 = self.analyzer.predicted_stack[:, lat, self.mlt_range[0]:]
                    segment2 = self.analyzer.predicted_stack[:, lat, :self.mlt_range[1]]
                    predicted = np.mean(np.concatenate([segment1, segment2], axis=1), axis=1)
                else:
                    # Normal range
                    ampere = np.mean(self.analyzer.ampere_stack[:, lat, self.mlt_slice], axis=1)
                    predicted = np.mean(self.analyzer.predicted_stack[:, lat, self.mlt_slice], axis=1)
            else:
                ampere = self.analyzer.ampere_stack[:, lat, self.mlt_slice]
                predicted = self.analyzer.predicted_stack[:, lat, self.mlt_slice]

            rolling_corrs_all[lat, :] = self.analyzer._rolling_corr_numba(ampere, predicted, window_size)

        rolling_times = self.analyzer.timestamps[window_size//2 : window_size//2 + n_windows]
        plot_times = self.analyzer._get_plot_timestamps(rolling_times)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

        time_indices = np.arange(len(plot_times))
        lat_indices = np.arange(self.analyzer.n_lats)

        im = ax1.pcolormesh(time_indices, lat_indices, rolling_corrs_all,
                           cmap='RdBu_r', vmin=-1, vmax=1, shading='auto',
                           rasterized=True)
        ax1.invert_yaxis()

        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        n_ticks = min(8, len(plot_times))
        tick_indices = np.linspace(0, len(plot_times)-1, n_ticks, dtype=int)
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([plot_times[i].strftime('%H:%M') if hasattr(plot_times[i], 'strftime')
                            else str(plot_times[i]) for i in tick_indices], rotation=45)

        lat_tick_indices = np.arange(0, self.analyzer.n_lats, max(1, self.analyzer.n_lats//10))
        ax1.set_yticks(lat_tick_indices)
        ax1.set_yticklabels([f'{self.analyzer.colat_to_lat(i)}°' for i in lat_tick_indices])

        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Latitude', fontsize=11)

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
            else:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
        else:
            mlt_str = f"MLT={self.mlt_idx}"

        ax1.set_title(f'Rolling Correlation (window={window_size}, {mlt_str})',
                     fontsize=12, fontweight='bold')

        mean_corr = rolling_corrs_all.mean(axis=1)
        std_corr = rolling_corrs_all.std(axis=1)
        lats = np.array([self.analyzer.colat_to_lat(i) for i in range(self.analyzer.n_lats)])

        ax2.plot(lats, mean_corr, 'b-', linewidth=2, label='Mean')
        ax2.fill_between(lats, mean_corr - std_corr, mean_corr + std_corr,
                         alpha=0.3, label='±1 std')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Latitude (°)', fontsize=11)
        ax2.set_ylabel('Mean Correlation', fontsize=11)
        ax2.set_title('Average Rolling Correlation by Latitude', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        overall_mean_corr = np.mean(rolling_corrs_all)
        min_corr = np.min(rolling_corrs_all)
        max_corr = np.max(rolling_corrs_all)
        temporal_variance = np.var(rolling_corrs_all.mean(axis=0))

        self.metrics['rolling_correlation'] = {
            'mean': overall_mean_corr,
            'min': min_corr,
            'max': max_corr,
            'temporal_variance': temporal_variance
        }

        self.add_summary_statement(
            f"Rolling correlation analysis (window={window_size}) shows mean correlation of {overall_mean_corr:.3f} "
            f"with range [{min_corr:.3f}, {max_corr:.3f}]. "
            f"{'High temporal variance indicates condition-dependent performance.' if temporal_variance > 0.05 else 'Stable performance across time.'}",
            category="Time-Varying Performance"
        )

    def run_spectral_analysis(self, pdf):
        """Run and document spectral analysis."""
        print("Running spectral analysis...")

        from scipy.fft import rfft, rfftfreq

        # Extract data using helper method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)

        n = len(ampere)

        fft_ampere = rfft(ampere)
        fft_pred = rfft(predicted)

        psd_ampere = np.abs(fft_ampere)**2 / n
        psd_pred = np.abs(fft_pred)**2 / n

        freq = rfftfreq(n, d=1.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(freq[1:], psd_ampere[1:], label='AMPERE', linewidth=2)
        ax.semilogy(freq[1:], psd_pred[1:], label='Predicted', linewidth=2)
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Power Spectral Density', fontsize=11)
        ax.set_title(f'Frequency Domain Analysis ({self._get_location_string()})',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self._save_figure(pdf, fig)

        ampere_peak_idx = np.argmax(psd_ampere[1:]) + 1
        pred_peak_idx = np.argmax(psd_pred[1:]) + 1

        self.metrics['spectral'] = {
            'ampere_peak_freq': freq[ampere_peak_idx],
            'predicted_peak_freq': freq[pred_peak_idx]
        }

        self.add_summary_statement(
            f"Spectral analysis shows dominant frequency at {freq[ampere_peak_idx]:.4f} Hz in observations "
            f"and {freq[pred_peak_idx]:.4f} Hz in predictions. "
            f"{'Frequencies align well.' if abs(freq[ampere_peak_idx] - freq[pred_peak_idx]) < 0.01 else 'Frequency mismatch detected.'}",
            category="Temporal Dynamics"
        )

    def run_autocorrelation_analysis(self, pdf):
        """Run and document autocorrelation analysis."""
        print("Running autocorrelation analysis...")

        # Extract data using helper method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)

        nlags = 20
        acf_ampere = self.analyzer._autocorr_numba(ampere, nlags)
        acf_predicted = self.analyzer._autocorr_numba(predicted, nlags)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.stem(range(nlags+1), acf_ampere)
        ax1.set_xlabel('Lag', fontsize=11)
        ax1.set_ylabel('Autocorrelation', fontsize=11)
        ax1.set_title(f'AMPERE Autocorrelation ({self._get_location_string()})',
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.stem(range(nlags+1), acf_predicted)
        ax2.set_xlabel('Lag', fontsize=11)
        ax2.set_ylabel('Autocorrelation', fontsize=11)
        ax2.set_title(f'Predicted Autocorrelation ({self._get_location_string()})',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        ampere_decorr = np.where(acf_ampere < 0.37)[0]
        pred_decorr = np.where(acf_predicted < 0.37)[0]

        ampere_decorr_lag = ampere_decorr[0] if len(ampere_decorr) > 0 else nlags
        pred_decorr_lag = pred_decorr[0] if len(pred_decorr) > 0 else nlags

        self.metrics['autocorrelation'] = {
            'ampere_decorr_lag': ampere_decorr_lag,
            'predicted_decorr_lag': pred_decorr_lag
        }

        self.add_summary_statement(
            f"Autocorrelation analysis shows decorrelation timescales of {ampere_decorr_lag} timesteps "
            f"for observations and {pred_decorr_lag} timesteps for predictions. "
            f"{'Model captures temporal persistence well.' if abs(ampere_decorr_lag - pred_decorr_lag) <= 2 else 'Temporal persistence mismatch detected.'}",
            category="Temporal Dynamics"
        )

    def run_hovmoller_analysis(self, pdf):
        """Run and document Hovmöller diagram."""
        print("Running Hovmöller analysis...")

        # For Hovmöller, average over MLT range if specified
        if self.mlt_mode == 'range':
            # Check for wraparound
            if self.mlt_range[0] > self.mlt_range[1]:
                # Wraparound
                segment1 = self.analyzer.ampere_stack[:, :, self.mlt_range[0]:]
                segment2 = self.analyzer.ampere_stack[:, :, :self.mlt_range[1]]
                ampere_array = np.mean(np.concatenate([segment1, segment2], axis=2), axis=2).T

                segment1 = self.analyzer.predicted_stack[:, :, self.mlt_range[0]:]
                segment2 = self.analyzer.predicted_stack[:, :, :self.mlt_range[1]]
                predicted_array = np.mean(np.concatenate([segment1, segment2], axis=2), axis=2).T
            else:
                # Normal range
                ampere_array = np.mean(self.analyzer.ampere_stack[:, :, self.mlt_slice], axis=2).T
                predicted_array = np.mean(self.analyzer.predicted_stack[:, :, self.mlt_slice], axis=2).T
        else:
            ampere_array = self.analyzer.ampere_stack[:, :, self.mlt_slice].T
            predicted_array = self.analyzer.predicted_stack[:, :, self.mlt_slice].T

        diff = ampere_array - predicted_array

        vmax = max(np.abs(ampere_array).max(), np.abs(predicted_array).max())
        vmin = -vmax

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        im1 = ax1.imshow(ampere_array, aspect='auto', cmap='RdBu_r',
                         origin='upper', vmin=vmin, vmax=vmax, rasterized=True)
        ax1.set_xlabel('Time Index', fontsize=11)
        ax1.set_ylabel('Latitude Index', fontsize=11)

        if self.mlt_mode == 'range':
            if self.mlt_range[0] > self.mlt_range[1]:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
            else:
                mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
        else:
            mlt_str = f"MLT={self.mlt_idx}"

        ax1.set_title(f'AMPERE Hovmöller ({mlt_str})', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(predicted_array, aspect='auto', cmap='RdBu_r',
                         origin='upper', vmin=vmin, vmax=vmax, rasterized=True)
        ax2.set_xlabel('Time Index', fontsize=11)
        ax2.set_ylabel('Latitude Index', fontsize=11)
        ax2.set_title(f'Predicted Hovmöller ({mlt_str})', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r', origin='upper', rasterized=True)
        ax3.set_xlabel('Time Index', fontsize=11)
        ax3.set_ylabel('Latitude Index', fontsize=11)
        ax3.set_title('Difference (AMPERE - Predicted)', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        self.add_summary_statement(
            f"Hovmöller diagram at {mlt_str} reveals spatial-temporal evolution patterns. "
            f"Visual inspection indicates {'good' if np.mean(np.abs(diff)) < 0.3 * vmax else 'significant'} "
            f"prediction errors in feature tracking.",
            category="Spatial-Temporal Evolution"
        )

    def run_calibration_analysis(self, pdf):
        """Run and document calibration analysis."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running calibration analysis...")

        # Extract data using helper method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)
        pred_std = self._extract_data(self.analyzer.std_stack)

        residuals = ampere - predicted
        z_scores = residuals / pred_std

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        from scipy import stats
        ax1 = axes[0, 0]
        ax1.scatter(pred_std, np.abs(residuals), alpha=0.6, s=20, rasterized=True)
        ax1.plot([0, pred_std.max()], [0, pred_std.max()], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Predicted Std Dev', fontsize=10)
        ax1.set_ylabel('Absolute Residual', fontsize=10)
        ax1.set_title('Calibration: Uncertainty vs Error', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.hist(z_scores, bins=30, density=True, alpha=0.7, label='Observed')
        x = np.linspace(-4, 4, 100)
        ax2.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
        ax2.set_xlabel('Z-score', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Z-Score Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        confidence_levels = np.linspace(0, 3, 31)
        observed_coverage = []

        for conf in confidence_levels:
            within = np.abs(z_scores) <= conf
            observed_coverage.append(np.mean(within))

        expected_coverage = stats.norm.cdf(confidence_levels) - stats.norm.cdf(-confidence_levels)

        ax3.plot(expected_coverage, observed_coverage, 'b-', linewidth=2, label='Model')
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        ax3.set_xlabel('Expected Coverage', fontsize=10)
        ax3.set_ylabel('Observed Coverage', fontsize=10)
        ax3.set_title('Calibration Curve', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        stats.probplot(z_scores, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        within_1sigma = np.mean(np.abs(z_scores) <= 1)
        within_2sigma = np.mean(np.abs(z_scores) <= 2)
        within_3sigma = np.mean(np.abs(z_scores) <= 3)
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)

        self.metrics['calibration'] = {
            'within_1sigma': within_1sigma,
            'within_2sigma': within_2sigma,
            'within_3sigma': within_3sigma,
            'mean_z': mean_z,
            'std_z': std_z
        }

        cal_1sig_diff = abs(within_1sigma - 0.683)
        cal_2sig_diff = abs(within_2sigma - 0.954)

        if cal_1sig_diff < 0.05 and cal_2sig_diff < 0.05:
            cal_quality = "excellent"
        elif cal_1sig_diff < 0.1 and cal_2sig_diff < 0.1:
            cal_quality = "good"
        elif cal_1sig_diff < 0.15:
            cal_quality = "acceptable"
        else:
            cal_quality = "poor"

        if within_1sigma < 0.683:
            confidence_msg = "overconfident (uncertainties too narrow)"
        elif within_1sigma > 0.683:
            confidence_msg = "underconfident (uncertainties too wide)"
        else:
            confidence_msg = "well-calibrated"

        self.add_summary_statement(
            f"Uncertainty calibration is {cal_quality}: {within_1sigma:.1%} within 1σ (expected 68.3%), "
            f"{within_2sigma:.1%} within 2σ (expected 95.4%). Model is {confidence_msg}. "
            f"Z-score statistics: mean={mean_z:.3f} (ideal: 0), std={std_z:.3f} (ideal: 1).",
            category="Uncertainty Quality"
        )

    def run_crps_analysis(self, pdf):
        """Run and document CRPS analysis."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running CRPS analysis...")

        from scipy import stats

        def crps_gaussian(observation, mean, std):
            z = (observation - mean) / std
            crps = std * (z * (2 * stats.norm.cdf(z) - 1) +
                          2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
            return crps

        # Extract data using helper method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)
        pred_std = self._extract_data(self.analyzer.std_stack)

        crps_scores = np.array([crps_gaussian(obs, mu, sigma)
                                for obs, mu, sigma in zip(ampere, predicted, pred_std)])

        plot_times = self.analyzer._get_plot_timestamps()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

        ax1.plot(plot_times, crps_scores, linewidth=2)
        ax1.set_ylabel('CRPS', fontsize=11)
        ax1.set_title(f'CRPS Over Time ({self._get_location_string()})',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        mae = np.abs(ampere - predicted)
        ax2.plot(plot_times, mae, label='MAE', linewidth=2)
        ax2.plot(plot_times, crps_scores, label='CRPS', linewidth=2)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('MAE vs CRPS Comparison', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure(pdf, fig)

        mean_crps = np.mean(crps_scores)
        mean_mae = np.mean(mae)
        crps_mae_ratio = mean_crps / mean_mae

        self.metrics['crps'] = {
            'mean_crps': mean_crps,
            'mean_mae': mean_mae,
            'ratio': crps_mae_ratio
        }

        improvement = (1 - crps_mae_ratio) * 100

        if crps_mae_ratio < 1:
            value_msg = f"Uncertainty reduces error by {improvement:.1f}% compared to just using the mean"
        else:
            value_msg = "Uncertainty does not add value beyond point predictions"

        self.add_summary_statement(
            f"CRPS/MAE = {crps_mae_ratio:.2f}; {value_msg}. "
            f"Mean CRPS={mean_crps:.4f}, Mean MAE={mean_mae:.4f}.",
            category="Uncertainty Quality"
        )

    def run_uncertainty_error_analysis(self, pdf):
        """Run and document uncertainty-error relationship."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running uncertainty-error relationship analysis...")

        residuals = self.analyzer.ampere_stack - self.analyzer.predicted_stack
        abs_errors = np.abs(residuals)

        errors_flat = abs_errors.flatten()
        uncertainty_flat = self.analyzer.std_stack.flatten()

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        ax1 = axes[0, 0]
        ax1.hexbin(uncertainty_flat, errors_flat, gridsize=50, cmap='viridis', mincnt=1, rasterized=True)
        ax1.plot([0, uncertainty_flat.max()], [0, uncertainty_flat.max()],
                 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Predicted Uncertainty', fontsize=10)
        ax1.set_ylabel('Actual Error', fontsize=10)
        ax1.set_title('Uncertainty vs Error', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        n_bins = 20
        bins = np.linspace(0, np.percentile(uncertainty_flat, 95), n_bins)
        bin_indices = np.digitize(uncertainty_flat, bins)

        bin_means_uncertainty = []
        bin_means_error = []
        bin_std_error = []

        for i in range(1, n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 10:
                bin_means_uncertainty.append(np.mean(uncertainty_flat[mask]))
                bin_means_error.append(np.mean(errors_flat[mask]))
                bin_std_error.append(np.std(errors_flat[mask]))

        if len(bin_means_uncertainty) > 0:
            bin_means_uncertainty = np.array(bin_means_uncertainty)
            bin_means_error = np.array(bin_means_error)
            bin_std_error = np.array(bin_std_error)

            ax2.errorbar(bin_means_uncertainty, bin_means_error, yerr=bin_std_error,
                         fmt='o-', linewidth=2, capsize=5, label='Observed')
            ax2.plot([0, bin_means_uncertainty.max()], [0, bin_means_uncertainty.max()],
                     'r--', linewidth=2, label='Perfect')

        ax2.set_xlabel('Mean Predicted Uncertainty', fontsize=10)
        ax2.set_ylabel('Mean Actual Error', fontsize=10)
        ax2.set_title('Binned Analysis', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]

        corr_map = np.zeros((self.analyzer.n_lats, self.analyzer.n_mlts))
        for i in range(self.analyzer.n_lats):
            for j in range(self.analyzer.n_mlts):
                errors = abs_errors[:, i, j]
                uncert = self.analyzer.std_stack[:, i, j]
                if np.std(errors) > 0 and np.std(uncert) > 0:
                    corr_map[i, j] = np.corrcoef(errors, uncert)[0, 1]

        im3 = ax3.imshow(corr_map, cmap='RdBu_r', vmin=-1, vmax=1,
                        origin='upper', aspect='auto', rasterized=True)

        # Draw boxes around pixels with correlation > 0.5
        from matplotlib.patches import Rectangle

        for i in range(self.analyzer.n_lats):
            for j in range(self.analyzer.n_mlts):
                if corr_map[i, j] > 0.7:
                    # Rectangle centered on pixel: (x, y, width, height)
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False,
                                edgecolor='green',
                                linewidth=2)
                    ax3.add_patch(rect)
                elif corr_map[i, j] > 0.5:
                    # Rectangle centered on pixel: (x, y, width, height)
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False,
                                edgecolor='blue',
                                linewidth=2)
                    ax3.add_patch(rect)

        ax3.set_title('Correlation by Location', fontsize=11, fontweight='bold')
        ax3.set_xlabel('MLT', fontsize=10)
        ax3.set_ylabel('Latitude', fontsize=10)
        plt.colorbar(im3, ax=ax3, label='Correlation')

        ax4 = axes[1, 1]
        ratio = errors_flat / (uncertainty_flat + 1e-10)
        ax4.hist(ratio, bins=50, density=True, alpha=0.7, range=(0, 5))
        ax4.axvline(x=1, color='r', linestyle='--', linewidth=2, label='Ratio=1')
        ax4.set_xlabel('Error / Uncertainty', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Error-to-Uncertainty Ratio', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        overall_corr = np.corrcoef(errors_flat, uncertainty_flat)[0, 1]
        median_ratio = np.median(ratio[ratio < 10])
        mean_spatial_corr = np.mean(corr_map)

        self.metrics['uncertainty_error'] = {
            'correlation': overall_corr,
            'median_ratio': median_ratio,
            'mean_spatial_corr': mean_spatial_corr
        }

        if median_ratio < 0.9:
            calibration_msg = "underconfident (uncertainties too large)"
        elif median_ratio > 1.1:
            calibration_msg = "overconfident (uncertainties too small)"
        else:
            calibration_msg = "well-calibrated"

        if overall_corr > 0.6:
            knowledge_msg = "Model demonstrates good awareness of when it is uncertain"
        elif overall_corr > 0.3:
            knowledge_msg = "Model shows moderate awareness of uncertainty patterns"
        else:
            knowledge_msg = "Model poorly estimates when predictions will be inaccurate"

        self.add_summary_statement(
            f"Uncertainty-error correlation is {overall_corr:.3f}, indicating "
            f"{'strong' if overall_corr > 0.6 else 'weak'} relationship. "
            f"Median error/uncertainty ratio is {median_ratio:.3f} ({calibration_msg}). "
            f"{knowledge_msg}.",
            category="Uncertainty Quality"
        )

    def run_sharpness_analysis(self, pdf):
        """Run and document sharpness analysis."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running sharpness analysis...")

        plot_times = self.analyzer._get_plot_timestamps()

        mean_std_time = np.mean(self.analyzer.std_stack, axis=(1, 2))
        median_std_time = np.median(self.analyzer.std_stack, axis=(1, 2))
        mean_std_spatial = np.mean(self.analyzer.std_stack, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        ax1 = axes[0, 0]
        ax1.plot(plot_times, mean_std_time, linewidth=2, label='Mean')
        ax1.plot(plot_times, median_std_time, linewidth=2, label='Median')
        ax1.set_xlabel('Time', fontsize=10)
        ax1.set_ylabel('Uncertainty (Std Dev)', fontsize=10)
        ax1.set_title('Sharpness Over Time', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        ax2 = axes[0, 1]
        im2 = ax2.imshow(mean_std_spatial, cmap='viridis', origin='upper', aspect='auto', rasterized=True)
        ax2.set_title('Mean Uncertainty by Location', fontsize=11, fontweight='bold')
        ax2.set_xlabel('MLT', fontsize=10)
        ax2.set_ylabel('Latitude', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='Std Dev')

        ax3 = axes[1, 0]
        ax3.hist(self.analyzer.std_stack.flatten(), bins=50, density=True, alpha=0.7)
        ax3.set_xlabel('Predicted Std Dev', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Distribution of Uncertainties', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        cv_time = np.std(self.analyzer.std_stack, axis=(1, 2)) / (mean_std_time + 1e-10)
        ax4.plot(plot_times, cv_time, linewidth=2)
        ax4.set_xlabel('Time', fontsize=10)
        ax4.set_ylabel('Coefficient of Variation', fontsize=10)
        ax4.set_title('Spatial Variability of Uncertainty', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        overall_mean = np.mean(self.analyzer.std_stack)
        overall_median = np.median(self.analyzer.std_stack)
        temporal_cv = np.std(mean_std_time) / np.mean(mean_std_time)
        spatial_cv = np.std(mean_std_spatial) / np.mean(mean_std_spatial)

        self.metrics['sharpness'] = {
            'mean': overall_mean,
            'median': overall_median,
            'temporal_cv': temporal_cv,
            'spatial_cv': spatial_cv
        }

        self.add_summary_statement(
            f"Model sharpness: mean uncertainty={overall_mean:.4f}, median={overall_median:.4f}. "
            f"Temporal variability (CV={temporal_cv:.3f}) "
            f"{'suggests condition-dependent uncertainty' if temporal_cv > 0.2 else 'is relatively stable'}. "
            f"Spatial variability (CV={spatial_cv:.3f}) "
            f"{'indicates location-dependent confidence' if spatial_cv > 0.2 else 'is uniform across domain'}.",
            category="Uncertainty Characteristics"
        )

    def run_probabilistic_skill_analysis(self, pdf):
        """Run and document probabilistic skill scores."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running probabilistic skill scores analysis...")

        from scipy import stats

        threshold = np.percentile(self.analyzer.ampere_stack, 90)

        z_threshold = (threshold - self.analyzer.predicted_stack) / self.analyzer.std_stack
        prob_exceed = 1 - stats.norm.cdf(z_threshold)

        observed_exceed = (self.analyzer.ampere_stack > threshold).astype(float)

        brier_score = np.mean((prob_exceed - observed_exceed)**2)

        climatology_prob = np.mean(observed_exceed)
        brier_score_clim = np.mean((climatology_prob - observed_exceed)**2)
        brier_skill_score = 1 - brier_score / brier_score_clim

        n_bins = 10
        prob_bins = np.linspace(0, 1, n_bins + 1)

        observed_freq = []
        forecast_prob = []
        counts = []

        for i in range(n_bins):
            mask = (prob_exceed >= prob_bins[i]) & (prob_exceed < prob_bins[i+1])
            if np.sum(mask) > 0:
                observed_freq.append(np.mean(observed_exceed[mask]))
                forecast_prob.append(np.mean(prob_exceed[mask]))
                counts.append(np.sum(mask))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax1 = axes[0]
        if len(forecast_prob) > 0:
            sizes = np.array(counts) / np.sum(counts) * 1000
            ax1.scatter(forecast_prob, observed_freq, s=sizes, alpha=0.6, rasterized=True)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Forecast Probability', fontsize=10)
        ax1.set_ylabel('Observed Frequency', fontsize=10)
        ax1.set_title(f'Reliability (Threshold={threshold:.2f})', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        ax2 = axes[1]
        sample_indices = np.random.choice(prob_exceed.size, min(5000, prob_exceed.size), replace=False)
        ax2.scatter(prob_exceed.flatten()[sample_indices],
                   self.analyzer.ampere_stack.flatten()[sample_indices],
                   alpha=0.3, s=1, rasterized=True)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Forecast Probability', fontsize=10)
        ax2.set_ylabel(r'Actual Current Density $\mu$A/$m^2$', fontsize=10)
        ax2.set_title('Probability vs Observations', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.hist(prob_exceed.flatten(), bins=30, density=True, alpha=0.7)
        ax3.axvline(x=climatology_prob, color='r', linestyle='--',
                   linewidth=2, label=f'Climatology ({climatology_prob:.2%})')
        ax3.set_xlabel('Forecast Probability', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Forecast Distribution', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(pdf, fig)

        self.metrics['probabilistic_skill'] = {
            'brier_score': brier_score,
            'brier_skill_score': brier_skill_score,
            'climatology_prob': climatology_prob,
            'threshold': threshold
        }

        if brier_skill_score > 0.3:
            skill_quality = "excellent"
        elif brier_skill_score > 0.1:
            skill_quality = "good"
        elif brier_skill_score > 0:
            skill_quality = "positive"
        else:
            skill_quality = "no better than climatology"

        self.add_summary_statement(
            f"Probabilistic forecasting skill for threshold exceedance (90th percentile = {threshold:.3f}): "
            f"Brier Score={brier_score:.4f}, Brier Skill Score={brier_skill_score:.4f} ({skill_quality}). "
            f"Model {'outperforms' if brier_skill_score > 0 else 'does not outperform'} climatology baseline.",
            category="Event Prediction"
        )

    def run_pseudo_ensemble_analysis(self, pdf):
        """Run and document pseudo-ensemble analysis."""
        if not self.analyzer.has_uncertainty:
            return

        print("Running pseudo-ensemble analysis...")

        # Extract data using helper method
        ampere = self._extract_data(self.analyzer.ampere_stack)
        predicted = self._extract_data(self.analyzer.predicted_stack)
        pred_std = self._extract_data(self.analyzer.std_stack)

        n_samples = 50
        n_times = len(ampere)

        ensemble = np.random.normal(
            loc=predicted[:, np.newaxis],
            scale=pred_std[:, np.newaxis],
            size=(n_times, n_samples)
        )

        plot_times = self.analyzer._get_plot_timestamps()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        ax1 = axes[0]
        for i in range(min(20, n_samples)):
            ax1.plot(plot_times, ensemble[:, i], 'b-', alpha=0.2, linewidth=0.5)
        ax1.plot(plot_times, predicted, 'b-', linewidth=2, label='Ensemble mean')
        ax1.plot(plot_times, ampere, 'k-', linewidth=2, label='Observed')
        ax1.fill_between(plot_times,
                         np.percentile(ensemble, 25, axis=1),
                         np.percentile(ensemble, 75, axis=1),
                         alpha=0.3, color='blue', label='IQR')
        ax1.set_ylabel(r'Current Density $\mu$A/$m^2$', fontsize=11)
        ax1.set_title(f'Pseudo-Ensemble Forecast ({self._get_location_string()})',
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ranks = np.zeros(n_times, dtype=int)
        for t in range(n_times):
            ranks[t] = np.sum(ensemble[t, :] < ampere[t])

        ax2.hist(ranks, bins=np.arange(n_samples + 2) - 0.5, density=True, alpha=0.7)
        ax2.axhline(y=1/(n_samples+1), color='r', linestyle='--',
                   linewidth=2, label='Uniform (perfect)')
        ax2.set_xlabel('Rank of Observation', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Rank Histogram (Should be Uniform)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ensemble_spread = np.std(ensemble, axis=1)
        errors = np.abs(ampere - predicted)

        ax3.scatter(ensemble_spread, errors, alpha=0.6, rasterized=True)
        max_val = max(ensemble_spread.max(), errors.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect spread-skill')
        ax3.set_xlabel('Ensemble Spread (Std Dev)', fontsize=11)
        ax3.set_ylabel('Forecast Error (Absolute)', fontsize=11)
        ax3.set_title('Spread-Skill Relationship', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure(pdf, fig)

        from scipy import stats
        spread_skill_ratio = np.mean(ensemble_spread) / np.mean(errors)
        rank_uniformity = stats.chisquare(np.bincount(ranks, minlength=n_samples+1))[1]

        self.metrics['pseudo_ensemble'] = {
            'spread_skill_ratio': spread_skill_ratio,
            'rank_uniformity_pvalue': rank_uniformity
        }

        self.add_summary_statement(
            f"Pseudo-ensemble analysis shows spread-skill ratio of {spread_skill_ratio:.3f} (ideal: 1.0). "
            f"Rank histogram uniformity p-value: {rank_uniformity:.4f}. "
            f"{'Ensemble is well-dispersed.' if 0.8 < spread_skill_ratio < 1.2 else 'Ensemble dispersion needs adjustment.'}",
            category="Ensemble Verification"
        )

    def create_metrics_summary_page(self, pdf):
        """Create a page summarizing all numerical metrics."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.98, 'Numerical Metrics Summary',
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        y_pos = 0.92
        line_height = 0.03

        ax.text(0.05, y_pos, 'Deterministic Performance:',
                ha='left', va='top', fontsize=13, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= line_height * 1.5

        if 'grid_cell' in self.metrics:
            m = self.metrics['grid_cell']
            metrics_text = [
                f"Point Correlation: {m['correlation']:.3f}",
                f"Point RMSE: {m['rmse']:.3f}",
                f"Point MAE: {m['mae']:.3f}",
                f"Point Bias: {m['bias']:.3f}"
            ]
            if 'uncertainty_error_corr' in m:
                metrics_text.append(f"Uncertainty-Error Corr: {m['uncertainty_error_corr']:.3f}")

            for text in metrics_text:
                ax.text(0.1, y_pos, text, ha='left', va='top', fontsize=10,
                        transform=ax.transAxes)
                y_pos -= line_height

        y_pos -= line_height * 0.5

        if 'cross_correlation' in self.metrics:
            m = self.metrics['cross_correlation']
            ax.text(0.1, y_pos, f"Optimal Lag: {m['optimal_lag']} timesteps",
                    ha='left', va='top', fontsize=10, transform=ax.transAxes)
            y_pos -= line_height
            ax.text(0.1, y_pos, f"Max Cross-Correlation: {m['max_correlation']:.3f}",
                    ha='left', va='top', fontsize=10, transform=ax.transAxes)
            y_pos -= line_height

        y_pos -= line_height * 0.5

        if 'rolling_correlation' in self.metrics:
            m = self.metrics['rolling_correlation']
            ax.text(0.1, y_pos, f"Mean Rolling Correlation: {m['mean']:.3f}",
                    ha='left', va='top', fontsize=10, transform=ax.transAxes)
            y_pos -= line_height

        if self.analyzer.has_uncertainty:
            y_pos -= line_height * 1.0
            ax.text(0.05, y_pos, 'Uncertainty Quantification:',
                    ha='left', va='top', fontsize=13, fontweight='bold',
                    transform=ax.transAxes)
            y_pos -= line_height * 1.5

            if 'calibration' in self.metrics:
                m = self.metrics['calibration']
                metrics_text = [
                    f"Coverage within 1σ: {m['within_1sigma']:.1%} (target: 68.3%)",
                    f"Coverage within 2σ: {m['within_2sigma']:.1%} (target: 95.4%)",
                    f"Z-score mean: {m['mean_z']:.3f} (target: 0)",
                    f"Z-score std: {m['std_z']:.3f} (target: 1)"
                ]
                for text in metrics_text:
                    ax.text(0.1, y_pos, text, ha='left', va='top', fontsize=10,
                            transform=ax.transAxes)
                    y_pos -= line_height

            y_pos -= line_height * 0.5

            if 'crps' in self.metrics:
                m = self.metrics['crps']
                ax.text(0.1, y_pos, f"Mean CRPS: {m['mean_crps']:.4f}",
                        ha='left', va='top', fontsize=10, transform=ax.transAxes)
                y_pos -= line_height
                ax.text(0.1, y_pos, f"CRPS/MAE Ratio: {m['ratio']:.3f}",
                        ha='left', va='top', fontsize=10, transform=ax.transAxes)
                y_pos -= line_height

            y_pos -= line_height * 0.5

            if 'uncertainty_error' in self.metrics:
                m = self.metrics['uncertainty_error']
                ax.text(0.1, y_pos, f"Uncertainty-Error Correlation: {m['correlation']:.3f}",
                        ha='left', va='top', fontsize=10, transform=ax.transAxes)
                y_pos -= line_height
                ax.text(0.1, y_pos, f"Median Error/Uncertainty Ratio: {m['median_ratio']:.3f}",
                        ha='left', va='top', fontsize=10, transform=ax.transAxes)
                y_pos -= line_height

            y_pos -= line_height * 0.5

            if 'probabilistic_skill' in self.metrics:
                m = self.metrics['probabilistic_skill']
                ax.text(0.1, y_pos, f"Brier Skill Score: {m['brier_skill_score']:.4f}",
                        ha='left', va='top', fontsize=10, transform=ax.transAxes)
                y_pos -= line_height

        self._save_figure(pdf, fig)

    def generate_report(self):
        """Generate the complete PDF report."""
        print(f"\nGenerating comprehensive analysis report: {self.output_filename}")
        print(f"Settings: DPI={self.dpi}, Compression={'ON' if self.compress_images else 'OFF'}")
        print("="*70)

        with PdfPages(self.output_filename) as pdf:
            print("Creating title page...")
            self.create_title_page(pdf)

            print("\n" + "="*70)
            print("DETERMINISTIC ANALYSES")
            print("="*70)

            self.run_grid_cell_analysis(pdf)
            self.run_regional_analysis(pdf)
            self.run_cross_correlation_analysis(pdf)
            # self.run_rolling_correlation_analysis(pdf)
            # self.run_spectral_analysis(pdf)
            self.run_autocorrelation_analysis(pdf)
            # self.run_hovmoller_analysis(pdf)

            if self.analyzer.has_uncertainty:
                print("\n" + "="*70)
                print("UNCERTAINTY-AWARE ANALYSES")
                print("="*70)

                self.run_calibration_analysis(pdf)
                self.run_crps_analysis(pdf)
                self.run_uncertainty_error_analysis(pdf)
                self.run_sharpness_analysis(pdf)
                # self.run_probabilistic_skill_analysis(pdf)
                # self.run_pseudo_ensemble_analysis(pdf)

            print("\nCreating metrics summary page...")
            self.create_metrics_summary_page(pdf)

            print("Creating executive summary...")
            self.create_executive_summary_page(pdf)

            d = pdf.infodict()
            d['Title'] = 'Model Analysis Report'
            d['Author'] = 'PostProcessingAnalysis'
            d['Subject'] = 'Geomagnetic Model Validation'
            d['Keywords'] = 'Model Validation, Statistics, Uncertainty Quantification'
            d['CreationDate'] = datetime.now()

        print("\n" + "="*70)
        print(f"Report generated successfully: {self.output_filename}")
        print("="*70)

        return self.output_filename


def main():
    """Example usage of the report generator."""

    version = 'ACORN_0_0'
    with open(os.path.expanduser(f'outputs/FAC_{version}_next_storm_training_False_results.pkl'), 'rb') as f:
        results_dict = pickle.load(f)

    # version for the UNet
    # for key in tqdm.tqdm(results_dict.keys()):
    #     results_dict[key]['std'] = pd.DataFrame(results_dict[key]['predicted'][1,:,:].reshape(50,24), columns=[i for i in range(24)], index=[i for i in range(50)])
    #     results_dict[key]['predicted'] = pd.DataFrame(results_dict[key]['predicted'][0,:,:].reshape(50,24), columns=[i for i in range(24)], index=[i for i in range(50)])
    #     results_dict[key]['ampere'] = pd.DataFrame(results_dict[key]['ampere'], columns=[i for i in range(24)], index=[i for i in range(50)])

    # version for the BK model
    for key in tqdm.tqdm(results_dict.keys()):
        results_dict[key]['std'] = pd.DataFrame(results_dict[key]['predicted'][1,:,:].reshape(50,24), columns=[i for i in range(24)], index=[i for i in range(50)])
        results_dict[key]['predicted'] = pd.DataFrame(results_dict[key]['predicted'][0,:,:].reshape(50,24), columns=[i for i in range(24)], index=[i for i in range(50)])
        results_dict[key]['ampere'] = pd.DataFrame(results_dict[key]['ampere'], columns=[i for i in range(24)], index=[i for i in range(50)])

    plot_time_range = ["2023-05-04 00:00:00", "2023-05-09 00:00:00"]
    time_range=None

    # Initialize analyzer
    analyzer = PostProcessingAnalysis(
        results_dict,
        time_range=time_range,  # optional
        timestamp_format='%Y-%m-%d %H:%M:%S'  # optional
    )
    # mlt_idx = 6
    # mlt_range = None

    mlt_idx = None
    mlt_range = (9,15)

    # lat_idx = 70
    # lat_range = None

    lat_idx = None
    lat_range = (65,75)

    mlt_title = mlt_idx if mlt_idx!=None else f'{mlt_range[0]}-{mlt_range[1]}'
    lat_title = lat_idx if lat_idx!=None else f'{lat_range[0]}-{lat_range[1]}'

    # Generate report
    report_gen = AnalysisReportGenerator(
        analyzer,
        output_filename=f'model_reports/{version}_mlt_{mlt_title}_lat_{lat_title}_model_report.pdf',
        lat_idx=lat_idx,
        mlt_idx=mlt_idx,
        lat_range=lat_range,
        mlt_range=mlt_range,
        plot_time_range=plot_time_range
    )

    report_filename = report_gen.generate_report()
    print(f"Report saved to: {report_filename}")


if __name__ == "__main__":
    main()
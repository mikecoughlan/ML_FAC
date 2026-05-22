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
            self.mlt_slice = slice(mlt_range[0], mlt_range[1])
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

    def _get_location_string(self):
        """Generate a string describing the analysis location."""
        if self.lat_mode == 'range':
            lat_str = f"Lat {self.lat_range[0]}-{self.lat_range[1]}°"
        else:
            lat_str = f"Lat={self.lat_idx}°"

        if self.mlt_mode == 'range':
            mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}"
        else:
            mlt_str = f"MLT={self.mlt_idx}"

        return f"{lat_str}, {mlt_str}"

    def _extract_data(self, data_array):
        """
        Extract and potentially average data based on lat/mlt mode.

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

        # Extract spatial subset
        if self.lat_mode == 'range' and self.mlt_mode == 'range':
            # Both ranges: extract and average over both dimensions
            subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(subset, axis=(1, 2))

        elif self.lat_mode == 'range' and self.mlt_mode == 'single':
            # Lat range, single MLT: extract MLT column and average over latitudes
            subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(subset, axis=1)

        elif self.lat_mode == 'single' and self.mlt_mode == 'range':
            # Single lat, MLT range: extract lat row and average over MLTs
            subset = data_array[:, self.colat_slice, self.mlt_slice]
            return np.mean(subset, axis=1)

        else:
            # Both single: extract single point
            return data_array[:, self.colat_slice, self.mlt_slice]

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

            ax1.set_ylabel('Value', fontsize=11)
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
            ax1.set_ylabel('Value', fontsize=11)
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
            # Average over MLT range
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
        cbar1.set_label('Value', rotation=270, labelpad=15)
        ax1.set_ylabel('Latitude', fontsize=11)

        if self.mlt_mode == 'range':
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
        cbar2.set_label('Value', rotation=270, labelpad=15)
        ax2.set_ylabel('Latitude', fontsize=11)

        if self.mlt_mode == 'range':
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

        mlt_str = f"MLT {self.mlt_range[0]}-{self.mlt_range[1]}" if self.mlt_mode == 'range' else f"MLT={self.mlt_idx}"

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
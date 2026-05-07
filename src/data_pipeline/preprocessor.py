import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


class Preprocessor:
    def __init__(self, outlier_sigma=5.0, savgol_window=15, savgol_order=3,
                 normalization_type='zscore'):  # <--- Αλλαγή σε 'zscore' ως default
        self.outlier_sigma = outlier_sigma
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.normalization_type = normalization_type

    def remove_outliers(self, time, flux, flux_err=None):
        # 1. Υπολογισμός baseline με median filter
        baseline = median_filter(flux, size=51)
        residuals = flux - baseline

        # Υπολογισμός MAD (Median Absolute Deviation) για στιβαρή εκτίμηση του θορύβου
        mad = np.median(np.abs(residuals)) * 1.4826
        if mad < 1e-9: mad = 1e-9

        # Ασύμμετρο φιλτράρισμα: Κόβουμε flares (+5 sigma), κρατάμε transits (-20 sigma)
        upper_limit = self.outlier_sigma * mad
        lower_limit = -20.0 * mad

        mask = (residuals < upper_limit) & (residuals > lower_limit)
        return mask

    def detrend(self, time, flux):
        window = self.savgol_window
        if len(flux) < 3:
            return flux - np.median(flux), np.full_like(flux, np.median(flux))

        if len(flux) <= window:
            window = len(flux)
            if window % 2 == 0: window -= 1

        if window < 3: window = 3

        try:
            trend = savgol_filter(flux, window, self.savgol_order)
            detrended_flux = flux - trend
        except Exception as e:
            trend = np.full_like(flux, np.median(flux))
            detrended_flux = flux - trend

        return detrended_flux, trend

    def normalize(self, flux):
        """
        Εφαρμογή Normalization.
        Επιλέγουμε Z-score για να αποφύγουμε το 'τεντωμα' του θορύβου.
        """
        if self.normalization_type == 'zscore':
            mean_f = np.mean(flux)
            std_f = np.std(flux)
            # Αποφυγή διαίρεσης με το μηδέν (Division by zero)
            return (flux - mean_f) / (std_f + 1e-9)

        elif self.normalization_type == 'median':
            return flux / (np.median(flux) + 1e-9)

        elif self.normalization_type == 'minmax':
            min_f, max_f = np.min(flux), np.max(flux)
            return (flux - min_f) / (max_f - min_f + 1e-9)
        else:
            raise ValueError(f"Unsupported normalization type: {self.normalization_type}")

    def process(self, time, flux, flux_err=None):
        # 0. Καθαρισμός NaNs και Infs
        valid_mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isinf(flux)
        time, flux = time[valid_mask], flux[mask := valid_mask]

        if flux_err is not None:
            flux_err = flux_err[valid_mask]

        # 1. Αφαίρεση outliers
        mask = self.remove_outliers(time, flux)
        time_clean, flux_clean = time[mask], flux[mask]

        if flux_err is not None:
            flux_err_clean = flux_err[mask]
        else:
            flux_err_clean = None

        # 2. Detrending (Αφαίρεση τάσης)
        flux_det, trend = self.detrend(time_clean, flux_clean)

        # 3. Normalization (Εδώ εφαρμόζεται το Z-score πλέον)
        flux_norm = self.normalize(flux_det)

        metadata = {
            'original_length': len(flux),
            'outliers_removed': int(np.sum(~mask)),
            'normalization': self.normalization_type
        }
        return time_clean, flux_norm, flux_err_clean, metadata
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


class Preprocessor:
    def __init__(self, outlier_sigma=5.0, savgol_window=15, savgol_order=3,
                 normalization_type='median'):
        self.outlier_sigma = outlier_sigma
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.normalization_type = normalization_type

    def remove_outliers(self, time, flux, flux_err=None):
        # 1. Υπολογισμός χοντρικού baseline
        # Αυξάνουμε το size (π.χ. 51) ώστε το median φίλτρο να είναι πιο "άκαμπτο"
        # και να ΜΗΝ βουτάει μέσα στο transit του πλανήτη.
        baseline = median_filter(flux, size=51)
        residuals = flux - baseline
        mad = np.median(np.abs(residuals)) * 1.4826

        # Προστασία σε περίπτωση που τα δεδομένα είναι αφύσικα τέλεια (π.χ. συνθετικά χωρίς θόρυβο)
        if mad < 1e-9:
            mad = 1e-9

        # 2. Ασύμμετρο Φιλτράρισμα (Asymmetric Sigma Clipping)
        # - Θετικά spikes (flares, cosmic rays): Αυστηρό όριο (π.χ. +5 sigma)
        upper_limit = self.outlier_sigma * mad

        # - Αρνητικά spikes (πλανήτες): Πολύ χαλαρό όριο (π.χ. -20 sigma)
        # Θέλουμε να κρατήσουμε τις βουτιές. Κόβουμε μόνο αν το σήμα
        # είναι εντελώς παράλογο (π.χ. σφάλμα αισθητήρα του δορυφόρου).
        lower_limit = -20.0 * mad

        mask = (residuals < upper_limit) & (residuals > lower_limit)
        return mask

    def detrend(self, time, flux):
        window = self.savgol_window

        # Αν τα δεδομένα είναι τραγικά λίγα, απλό median subtraction (Fallback)
        if len(flux) < 3:
            return flux - np.median(flux), np.full_like(flux, np.median(flux))

        # Δυναμική προσαρμογή παραθύρου
        if len(flux) <= window:
            window = len(flux)
            if window % 2 == 0:  # Πρέπει να είναι αυστηρά περιττός αριθμός
                window -= 1

        if window < 3:
            window = 3

        try:
            trend = savgol_filter(flux, window, self.savgol_order)
            detrended_flux = flux - trend
        except Exception as e:
            # Αν σκάσει για οποιονδήποτε λόγο (π.χ. μαθηματικό error του scipy)
            print(f"  [Warning] Savgol failed ({e}). Fallback to median detrending.")
            trend = np.full_like(flux, np.median(flux))
            detrended_flux = flux - trend

        return detrended_flux, trend

    def normalize(self, flux):
        if self.normalization_type == 'median':
            median_flux = np.median(flux)
            if median_flux == 0:
                raise ValueError("Median flux is zero.")
            return flux / median_flux
        elif self.normalization_type == 'minmax':
            min_f, max_f = np.min(flux), np.max(flux)
            return (flux - min_f) / (max_f - min_f + 1e-9)
        elif self.normalization_type == 'zscore':
            mean_f = np.mean(flux)
            std_f = np.std(flux)
            if std_f == 0:
                return flux - mean_f
            return (flux - mean_f) / std_f
        else:
            raise ValueError("Unsupported normalization type.")

    def process(self, time, flux, flux_err=None):
        # 0. Data Sanitization: Αφαίρεση NaNs και Infs
        valid_mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isinf(flux)
        time = time[valid_mask]
        flux = flux[valid_mask]
        if flux_err is not None:
            flux_err = flux_err[valid_mask]

        # 1. Αφαίρεση outliers
        mask = self.remove_outliers(time, flux, flux_err)
        time_clean = time[mask]
        flux_clean = flux[mask]

        # ... (το υπόλοιπο κομμάτι παραμένει ως έχει)
        if flux_err is not None:
            flux_err_clean = flux_err[mask]
        else:
            flux_err_clean = None

        flux_det, trend = self.detrend(time_clean, flux_clean)
        flux_norm = self.normalize(flux_det)

        metadata = {
            'original_length': len(flux),  # Τώρα μετράει το μήκος χωρίς τα NaNs
            'outliers_removed': int(np.sum(~mask)),
            'trend_mean': float(np.mean(trend))
        }
        return time_clean, flux_norm, flux_err_clean, metadata
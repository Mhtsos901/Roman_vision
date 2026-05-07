import numpy as np
from astropy.timeseries import BoxLeastSquares

class TransitDetector:
    def __init__(self, min_period=0.5, max_period=10.0,
                 min_duration=0.01, max_duration=0.1, n_durations=20):
        """
        Args:
            min_period, max_period: εύρος περιόδων (ημέρες).
            min_duration, max_duration: εύρος διάρκειας ως ποσοστό της περιόδου.
            n_durations: πόσες διαφορετικές διάρκειες θα δοκιμάσουμε.
        """
        self.min_period = min_period
        self.max_period = max_period
        self.durations = np.linspace(min_duration, max_duration, n_durations)

    def search(self, time, flux, flux_err=None):
        if flux_err is None:
            flux_err = np.ones_like(flux) * 0.001
        bls = BoxLeastSquares(time, flux, dy=flux_err)
        results = bls.autopower(self.durations, minimum_period=self.min_period,
                                maximum_period=self.max_period)
        best_idx = np.argmax(results.power)
        period = results.period[best_idx]
        power = results.power[best_idx]
        duration = results.duration[best_idx]
        return results, period, power, duration

    def search_top_n(self, time, flux, flux_err=None, n_top=5):
        """
        Ψάχνει για περιοδικές διαβάσεις και επιστρέφει τις n_top καλύτερες περιόδους,
        με τα αντίστοιχα power και duration.
        """
        if flux_err is None:
            flux_err = np.ones_like(flux) * 0.001
        bls = BoxLeastSquares(time, flux, dy=flux_err)
        results = bls.autopower(self.durations, minimum_period=self.min_period,
                                maximum_period=self.max_period)
        # Ταξινόμηση κατά φθίνουσα ισχύ
        idx_sorted = np.argsort(results.power)[::-1]
        top_periods = results.period[idx_sorted][:n_top]
        top_powers = results.power[idx_sorted][:n_top]
        top_durations = results.duration[idx_sorted][:n_top]
        return top_periods, top_powers, top_durations, results  # επιστρέφουμε και ολόκληρο το results για γράφημα

    def is_planet_like(self, power, threshold=5.0):
        return power > threshold
import numpy as np
from scipy.interpolate import interp1d
import lightkurve as lk
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

class ModelTrainer:
    def __init__(self, validator, preprocessor, input_length=201,
                 n_synthetic=800, n_real_positive=5):
        self.validator = validator
        self.preprocessor = preprocessor
        self.input_length = input_length
        self.n_synthetic = n_synthetic
        self.n_real_positive = n_real_positive   # πόσοι αληθινοί πλανήτες θα χρησιμοποιηθούν

    def _scale_to_zscore(self, flux_array):
        """
        Βοηθητική μέθοδος: Εφαρμόζει Z-score standardization (zero mean, unit variance).
        Αποτρέπει το Data Leakage ή το Distribution Shift μεταξύ Train και Inference.
        """
        mean_f = np.mean(flux_array)
        std_f = np.std(flux_array)

        # Προστασία από διαίρεση με το μηδέν (division by zero)
        if std_f > 1e-9:
            return (flux_array - mean_f) / std_f
        return flux_array - mean_f

    def _simulate_and_preprocess(self, depth=0.01, width=0.04, noise_level=0.002):
        time = np.linspace(0, 1, self.input_length * 2)
        white_noise = np.random.normal(0, noise_level, len(time))
        red_noise = np.cumsum(white_noise) * 0.1
        stellar_var = 0.005 * np.sin(2 * np.pi * time / 0.3)
        flux = 1.0 + red_noise + stellar_var + white_noise

        in_transit = np.abs(time - 0.5) < width
        flux[in_transit] -= depth

        time_clean, flux_clean, _, _ = self.preprocessor.process(time, flux, None)
        f = interp1d(time_clean, flux_clean, kind='linear', bounds_error=False, fill_value=0.0)

        # Global View
        global_grid = np.linspace(time_clean.min(), time_clean.max(), self.input_length)
        flux_global = self._scale_to_zscore(f(global_grid))

        # Local View
        mid_point = (time_clean.min() + time_clean.max()) / 2.0
        span = (time_clean.max() - time_clean.min()) * 0.1
        local_grid = np.linspace(mid_point - span, mid_point + span, self.input_length)
        flux_local = self._scale_to_zscore(f(local_grid))

        return flux_global, flux_local

    def _generate_noise_curve(self, noise_level=0.002):
        time = np.linspace(0, 1, self.input_length * 2)
        white_noise = np.random.normal(0, noise_level, len(time))
        red_noise = np.cumsum(white_noise) * 0.1
        stellar_var = 0.005 * np.sin(2 * np.pi * time / 0.3)
        flux = 1.0 + red_noise + stellar_var + white_noise

        time_clean, flux_clean, _, _ = self.preprocessor.process(time, flux, None)
        f = interp1d(time_clean, flux_clean, kind='linear', bounds_error=False, fill_value=0.0)

        # Global View
        global_grid = np.linspace(time_clean.min(), time_clean.max(), self.input_length)
        flux_global = self._scale_to_zscore(f(global_grid))

        # Local View
        mid_point = (time_clean.min() + time_clean.max()) / 2.0
        span = (time_clean.max() - time_clean.min()) * 0.1
        local_grid = np.linspace(mid_point - span, mid_point + span, self.input_length)
        flux_local = self._scale_to_zscore(f(local_grid))

        return flux_global, flux_local

    def _fetch_real_transit(self, tic, period, sector=None):
        try:
            search = lk.search_lightcurve(f"TIC {tic}", mission='TESS', author='SPOC', sector=sector)
            if len(search) == 0:
                return None
            lc = search.download()
            time_raw = lc.time.value
            flux_raw = lc.flux.value
            time_clean, flux_clean, _, _ = self.preprocessor.process(time_raw, flux_raw, None)
            phase, flux_folded = fold_lightcurve(time_clean, flux_clean, period)
            f = interp1d(phase, flux_folded, kind='linear', bounds_error=False, fill_value=0.0)

            # Global View
            global_grid = np.linspace(0, 1, self.input_length)
            flux_global = self._scale_to_zscore(f(global_grid))

            # Local View (Επειδή είναι folded, το κέντρο είναι στο 0.5)
            mid_point = 0.5
            span = 0.1
            local_grid = np.linspace(mid_point - span, mid_point + span, self.input_length)
            flux_local = self._scale_to_zscore(f(local_grid))

            return flux_global, flux_local
        except Exception as e:
            print(f"    Αποτυχία για TIC {tic}: {e}")
            return None

    def generate_training_data(self):
        X_pos, y_pos = [], []
        X_neg, y_neg = [], []

        # ----- 1. Συνθετικά θετικά -----
        print(f"Δημιουργία {self.n_synthetic} συνθετικών θετικών δειγμάτων...")
        for _ in range(self.n_synthetic):
            # Ρυθμισμένο για πολύ ρηχές (0.1%) έως κανονικές διαβάσεις (2%)
            depth = np.random.uniform(0.001, 0.02)
            width = np.random.uniform(0.02, 0.06)
            noise = np.random.uniform(0.001, 0.003)

            # Τώρα επιστρέφει Tuple: (flux_global, flux_local)
            sample = self._simulate_and_preprocess(depth=depth, width=width, noise_level=noise)
            X_pos.append(sample)
            y_pos.append(1)

        # ----- 2. Αληθινοί πλανήτες (θετικά) -----
        real_planets = [
            ("25155310", 0.941452),  # WASP-18 b (Test set - θα αφαιρεθεί)
            ("150428135", 3.484),  # TOI-1235 b
            ("307210830", 1.091),  # TOI-270 c
            ("34745214", 0.359),  # TOI-182 b
            ("200593988", 4.27),  # TOI-561 b
            ("408310944", 1.493),  # TOI-2000 b
            ("141527959", 1.274),  # TOI-362 b
            ("281653000", 1.393),  # TOI-1231 b
            ("157760092", 0.695),  # TOI-943 b
            ("445554663", 2.939),  # TOI-1726 b
        ]

        train_planets = [(tic, p) for (tic, p) in real_planets if tic != "25155310"]
        train_planets = train_planets[:self.n_real_positive]

        print(f"Λήψη {len(train_planets)} αληθινών πλανητών...")
        for tic, period in train_planets:
            print(f"  Δοκιμή TIC {tic} (P={period:.3f} d)...")
            sample = self._fetch_real_transit(tic, period)
            if sample is not None:
                # To sample είναι Tuple: (flux_global, flux_local)
                X_pos.append(sample)
                y_pos.append(1)
                print(f"    ✓ Προστέθηκε!")
            else:
                print(f"    ✗ Δεν βρέθηκε ή απέτυχε η επεξεργασία.")

        # ----- 3. Αρνητικά δείγματα (συνθετικός θόρυβος) -----
        print(f"Δημιουργία {self.n_synthetic} αρνητικών δειγμάτων (θόρυβος)...")
        for _ in range(self.n_synthetic):
            noise = np.random.uniform(0.001, 0.003)

            # Τώρα επιστρέφει Tuple: (flux_global, flux_local)
            sample = self._generate_noise_curve(noise_level=noise)
            X_neg.append(sample)
            y_neg.append(0)

        # ----- 4. Ενοποίηση και Προετοιμασία για το Multi-Input Keras Model -----
        print("Προετοιμασία των Global και Local Tensors...")

        # Διαχωρίζουμε τα tuples σε δύο ξεχωριστές λίστες
        # item[0] είναι το global_flux, item[1] είναι το local_flux
        X_global = np.array([item[0] for item in X_pos] + [item[0] for item in X_neg])
        X_local = np.array([item[1] for item in X_pos] + [item[1] for item in X_neg])
        y = np.concatenate([np.array(y_pos), np.array(y_neg)])

        # Reshape για το Conv1D (samples, timesteps, channels)
        X_global = X_global.reshape(-1, self.input_length, 1)
        X_local = X_local.reshape(-1, self.input_length, 1)

        # Ανακάτεμα (Shuffling)
        # ΠΡΟΣΟΧΗ: Πρέπει να ανακατέψουμε και τα 3 arrays με το ΙΔΙΟ index!
        idx = np.random.permutation(len(y))
        X_global = X_global[idx]
        X_local = X_local[idx]
        y = y[idx]

        print(f"Σύνολο δειγμάτων: {len(y)} (θετικά: {np.sum(y)}, αρνητικά: {len(y) - np.sum(y)})")

        # Επιστρέφουμε λίστα με τα X (όπως απαιτεί το functional API του Keras) και το y
        return [X_global, X_local], y

    def train(self, epochs=30):
        X, y = self.generate_training_data()
        history = self.validator.model.fit(
            X, y, epochs=epochs, validation_split=0.2, verbose=1
        )
        return history
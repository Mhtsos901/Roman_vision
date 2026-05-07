import numpy as np
import tensorflow as tf
import lightkurve as lk
import glob
from scipy.interpolate import interp1d

from src.models.planet_validator import PlanetValidator
from src.models.trainer import ModelTrainer
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

# =========================================================
# 1. ΕΚΠΑΙΔΕΥΣΗ ΜΕ ΡΕΑΛΙΣΤΙΚΕΣ (ΜΙΚΡΕΣ) ΒΟΥΤΙΕΣ
# =========================================================
print("===== ΕΚΠΑΙΔΕΥΣΗ ΜΕ ΡΕΑΛΙΣΤΙΚΑ ΔΕΔΟΜΕΝΑ =====")
validator = PlanetValidator(input_length=201)
prep = Preprocessor(outlier_sigma=5.0, savgol_window=21)
trainer = ModelTrainer(validator, prep, input_length=201, n_synthetic=800, n_real_positive=5)
history = trainer.train(epochs=50)

# =========================================================
# 2. ΔΟΚΙΜΗ ΣΕ ΠΡΑΓΜΑΤΙΚΑ ΔΕΔΟΜΕΝΑ (WASP-18 b)
# =========================================================
print("\n===== ΔΟΚΙΜΗ ΣΕ WASP-18 b =====")
tic_id = 25155310
period = 0.941452

# 2.1 Βρες το αρχείο (τοπικά ή κατέβασέ το)
local_files = glob.glob(f'data/raw/*{tic_id}*')
if local_files:
    print(f"Βρέθηκε τοπικό αρχείο: {local_files[0]}")
    lc = lk.read(local_files[0])
else:
    print("Αναζήτηση στο MAST...")
    search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')
    if len(search) == 0:
        print("Δεν βρέθηκε τίποτα. Χρήση συνθετικής βουτιάς για επίδειξη.")
        phase_grid = np.linspace(0, 1, 201)
        flux = np.ones(201)
        in_transit = np.abs(phase_grid - 0.5) < 0.04
        flux[in_transit] -= 0.01
        flux += np.random.normal(0, 0.0005, 201)
        mean_f = np.mean(flux)
        std_f = np.std(flux)
        if std_f > 1e-9:
            flux = (flux - mean_f) / std_f
        else:
            flux = flux - mean_f
        input_data = flux.reshape(1, 201, 1)
        prob = validator.model.predict(input_data, verbose=0)[0][0]
        # ...
        print(f"Πιθανότητα (συνθετική): {prob:.4f}")
        exit()
    lc = search.download()
    print("Επιτυχής λήψη!")

# 2.2 Προεπεξεργασία
prep = Preprocessor(outlier_sigma=5.0, savgol_window=21)
time, flux = lc.time.value, lc.flux.value
time_c, flux_c, _, _ = prep.process(time, flux)

# 2.3 Αναδίπλωση
phase, flux_folded = fold_lightcurve(time_c, flux_c, period)

# 2.4 & 2.5 Πρόβλεψη (χρησιμοποιώντας τη σωστή μέθοδο της κλάσης που τα κάνει όλα!)
prob = validator.predict(phase, flux_folded)

print(f"\n🔮 Πιθανότητα ύπαρξης πλανήτη (WASP-18 b): {prob:.4f}")
if prob > 0.5:
    print("   👉 Το CNN προβλέπει: ΠΙΘΑΝΟΣ ΕΞΩΠΛΑΝΗΤΗΣ!")
else:
    print("   👉 Το CNN προβλέπει: ΜΑΛΛΟΝ ΔΕΝ ΥΠΑΡΧΕΙ ΠΛΑΝΗΤΗΣ.")
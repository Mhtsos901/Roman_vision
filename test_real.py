import numpy as np
import tensorflow as tf
import lightkurve as lk
import glob
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve
from scipy.interpolate import interp1d

# 1. Φόρτωση μοντέλου
print("Φόρτωση μοντέλου...")
model = tf.keras.models.load_model('models/planet_validator.h5')
print("Μοντέλο έτοιμο.")

tic_id = 25155310
period = 0.941452

# 2. Προσπάθεια εύρεσης τοπικού αρχείου
local_files = glob.glob(f'data/raw/*{tic_id}*')
if local_files:
    print(f"Βρέθηκε τοπικό αρχείο: {local_files[0]}")
    lc = lk.read(local_files[0])
else:
    # 3. Αναζήτηση σε όλους τους sectors (χωρίς sector)
    print(f"Αναζήτηση όλων των sectors για TIC {tic_id}...")
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')
        if len(search) == 0:
            raise ValueError("Κανένα αποτέλεσμα.")
        print(f"Βρέθηκαν {len(search)} entries. Λήψη του πρώτου...")
        lc = search.download()  # κατεβάζει το πρώτο διαθέσιμο
        print("Επιτυχής λήψη!")
    except Exception as e:
        print(f"Σφάλμα λήψης: {e}")
        print("Χρησιμοποιώ συνθετική βουτιά.")
        # Συνθετική βουτιά
        phase_grid = np.linspace(0, 1, 201)
        flux = np.ones(201)
        in_transit = np.abs(phase_grid - 0.5) < 0.04
        flux[in_transit] -= 0.01
        flux += np.random.normal(0, 0.0005, 201)
        flux = (flux - flux.min()) / (flux.max() - flux.min())
        input_data = flux.reshape(1, 201, 1)
        prob = model.predict(input_data, verbose=0)[0][0]
        print(f"Πιθανότητα (συνθετική): {prob:.4f}")
        exit()

# 4. Προεπεξεργασία και αναδίπλωση
prep = Preprocessor(outlier_sigma=5.0, savgol_window=21)
time, flux = lc.time.value, lc.flux.value
time_c, flux_c, _, _ = prep.process(time, flux)
phase, flux_folded = fold_lightcurve(time_c, flux_c, period)

# 5. Προετοιμασία για το CNN (201 σημεία, min‑max)
f = interp1d(phase, flux_folded, kind='linear', bounds_error=False, fill_value=0.0)
phase_grid_cnn = np.linspace(0, 1, 201)
flux_cnn = f(phase_grid_cnn)
flux_cnn = (flux_cnn - flux_cnn.min()) / (flux_cnn.max() - flux_cnn.min() + 1e-9)
cnn_input = flux_cnn.reshape(1, 201, 1)

# 6. Πρόβλεψη
prob = model.predict(cnn_input, verbose=0)[0][0]
print(f"\n🔮 Πιθανότητα ύπαρξης πλανήτη (WASP-18 b): {prob:.4f}")
if prob > 0.5:
    print("   👉 Το CNN προβλέπει: ΠΙΘΑΝΟΣ ΕΞΩΠΛΑΝΗΤΗΣ!")
else:
    print("   👉 Το CNN προβλέπει: ΜΑΛΛΟΝ ΔΕΝ ΥΠΑΡΧΕΙ ΠΛΑΝΗΤΗΣ.")
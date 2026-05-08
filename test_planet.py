import os
import numpy as np
import tensorflow as tf
import lightkurve as lk
import warnings
from scipy.spatial import distance

from src.models.planet_validator import PlanetValidator
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

# Απόκρυψη warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ==========================================
# 1. ΠΑΡΑΜΕΤΡΟΙ ΣΤΟΧΟΥ & ΡΥΘΜΙΣΕΙΣ
# ==========================================
tic_id = 37749396  # Δοκίμασε όποιο TIC θέλεις
period = 0.941
threshold_dist = 1.0  # Το νέο "Χρυσό Όριο" για το Z-score

model_path = 'models/planet_validator_realistic.h5'
centroid_path = 'models/planet_centroid.npy'
# ==========================================

print(f"\n🚀 Ξεκινάει ο έλεγχος για το TIC {tic_id}")

# 2. Φόρτωση Μοντέλου και Centroid
print("📂 Φόρτωση μοντέλου και ψηφιακού προτύπου (Centroid)...")
validator = PlanetValidator(input_length=201)
validator.model = tf.keras.models.load_model(model_path)

if not os.path.exists(centroid_path):
    print("❌ ERROR: Το αρχείο Centroid δεν βρέθηκε!")
    exit()

planet_centroid = np.load(centroid_path, allow_pickle=True)

# 3. Λήψη Δεδομένων
print("📡 Λήψη δεδομένων από το TESS...")
search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')

if len(search) == 0:
    print(f"❌ Δεν βρέθηκαν δεδομένα.")
    exit()

try:
    lc = search[0].download()
    if lc is None:
        print("❌ Το download επέστρεψε κενό αρχείο.")
        exit()
except Exception as e:
    print(f"❌ Σφάλμα κατά τη λήψη: {e}")
    exit()

# Ασφαλές Casting για αποφυγή του MaskedNDArray bug
time = np.array(lc.time.value, dtype=np.float64)
flux = np.array(lc.flux.value, dtype=np.float64)

mask = ~np.isnan(flux)
time, flux = time[mask], flux[mask]

# 4. Preprocessing & Folding
print("🧼 Καθαρισμός και αναδίπλωση σήματος (Folding)...")
prep = Preprocessor(normalization_type='zscore')
time_clean, flux_norm, _, _ = prep.process(time, flux)
phase, flux_folded = fold_lightcurve(time_clean, flux_norm, period)

# 5. Dual-Gate Validation (Inference)
print("🧠 Εκτέλεση Inference και ανάλυση Latent Space...")

prob = float(np.squeeze(validator.predict(phase, flux_folded)))
current_features = validator.get_features(phase, flux_folded)
dist = float(distance.euclidean(current_features, planet_centroid))

# 6. Τελική Αναφορά
print("\n" + "="*45)
print(f"📊 ΣΤΑΤΙΣΤΙΚΗ ΑΝΑΛΥΣΗ ΓΙΑ TIC {tic_id}:")
print(f"🔮 Πιθανότητα (CNN): {prob:.4f} ({(prob*100):.2f}%)")
print(f"📏 Απόσταση (Euclidean): {dist:.4f}")
print("="*45)

# Καθαρή Λογική Απόφασης
if prob > 0.5:
    if dist <= threshold_dist:
        print("✅ ΑΠΟΤΕΛΕΣΜΑ: PLANET CANDIDATE")
        print("   (Το σήμα και η γεωμετρία ταιριάζουν με πλανήτη/έκλειψη)")
    else:
        print("❌ ΑΠΟΤΕΛΕΣΜΑ: ANOMALY (Noise/Artifact)")
        print("   (Υψηλή πιθανότητα, αλλά η γεωμετρία απέχει από το πρότυπο)")
else:
    print("🪨 ΑΠΟΤΕΛΕΣΜΑ: NO PLANET")

print("="*45 + "\n")
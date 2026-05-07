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

# ==========================================
# 1. ΠΑΡΑΜΕΤΡΟΙ ΣΤΟΧΟΥ & ΡΥΘΜΙΣΕΙΣ
# ==========================================
tic_id = 281541555
period = 1.4
threshold_dist = 1.5  # Καλιμπραρισμένο για Z-score
threshold_snr = 7.0  # Το επιστημονικό όριο για αξιόπιστο σήμα (NASA standard)

model_path = 'models/planet_validator_realistic.h5'
centroid_path = 'models/planet_centroid.npy'


# ==========================================

def calculate_snr(phase, flux):
    """
    Υπολογίζει το Signal-to-Noise Ratio της αναδιπλωμένης καμπύλης.
    """
    # Θεωρούμε ότι το transit είναι κεντραρισμένο στη φάση 0.5 (εύρος 0.45-0.55)
    transit_mask = (phase > 0.45) & (phase < 0.55)
    out_mask = ~transit_mask

    if not np.any(transit_mask) or not np.any(out_mask):
        return 0

    # Baseline (Επίπεδο φωτός άστρου) vs Transit Level (Βάθος βουτιάς)
    baseline = np.median(flux[out_mask])
    transit_level = np.median(flux[transit_mask])
    depth = baseline - transit_level

    # Θόρυβος (Standard Deviation) εκτός της περιοχής του transit
    noise_std = np.std(flux[out_mask])

    if noise_std == 0: return 0

    # SNR = (Depth / Noise) * sqrt(N_points_in_transit)
    n_points = np.sum(transit_mask)
    snr = (depth / noise_std) * np.sqrt(n_points)

    return snr


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

lc = search[0].download()
time, flux = lc.time.value, lc.flux.value

# Καθαρισμός NaNs (Απαραίτητο για Z-score / CNN)
mask = ~np.isnan(flux)
time, flux = time[mask], flux[mask]

# 4. Preprocessing & Folding
print("🧼 Καθαρισμός και αναδίπλωση σήματος (Folding)...")
prep = Preprocessor(normalization_type='zscore')  # Βεβαιώσου ότι είναι zscore
time_clean, flux_norm, _, _ = prep.process(time, flux)
phase, flux_folded = fold_lightcurve(time_clean, flux_norm, period)

# 5. Triple Validation Analysis
print("🧠 Εκτέλεση Inference και Στατιστική Ανάλυση...")

# Α) Πιθανότητα (CNN)
prob = validator.predict(phase, flux_folded)

# Β) Απόσταση (Embedding Distance)
current_features = validator.get_features(phase, flux_folded)
dist = distance.euclidean(current_features, planet_centroid)

# Γ) Signal-to-Noise Ratio (SNR)
snr_val = calculate_snr(phase, flux_folded)

# 6. Τελική Αναφορά
print("\n" + "=" * 45)
print(f"📊 ΣΤΑΤΙΣΤΙΚΗ ΑΝΑΛΥΣΗ ΓΙΑ TIC {tic_id}:")
print(f"🔮 Πιθανότητα (CNN): {prob:.4f} ({(prob * 100):.2f}%)")
print(f"📏 Απόσταση (Euclidean): {dist:.4f}")
print(f"📡 SNR: {snr_val:.2f}")
print("=" * 45)

# Λογική Απόφασης (The Senior Gatekeeper)
is_high_prob = prob > 0.5
is_low_dist = dist < threshold_dist
is_strong_signal = snr_val > threshold_snr

if is_high_prob and is_low_dist and is_strong_signal:
    print("✅ ΑΠΟΤΕΛΕΣΜΑ: ΠΙΘΑΝΟΣ ΕΞΩΠΛΑΝΗΤΗΣ (Σήμα Υψηλής Εμπιστοσύνης)")
elif is_high_prob and is_low_dist and not is_strong_signal:
    print("⚠️ ΑΠΟΤΕΛΕΣΜΑ: ΥΠΟΨΗΦΙΟΣ ΑΛΛΑ ΧΑΜΗΛΟ SNR")
    print("   (Το σχήμα μοιάζει σωστό, αλλά το σήμα είναι πολύ αδύναμο/θόρυβος)")
elif is_high_prob and not is_low_dist:
    print("❌ ΑΠΟΤΕΛΕΣΜΑ: ΑΝΩΜΑΛΙΑ / FALSE POSITIVE")
    print("   (Υψηλή πιθανότητα, αλλά η γεωμετρία (Latent Space) είναι λάθος)")
else:
    print("🪨 ΑΠΟΤΕΛΕΣΜΑ: ΜΑΛΛΟΝ ΔΕΝ ΥΠΑΡΧΕΙ ΠΛΑΝΗΤΗΣ")

print("=" * 45 + "\n")
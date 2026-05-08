import os
import numpy as np
import pandas as pd
import tensorflow as tf
import lightkurve as lk
import warnings
from scipy.spatial import distance

from src.models.planet_validator import PlanetValidator
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def run_batch_scan():
    print("🚀 Εκκίνηση TESS Batch Scanner (Dual-Gate AI Validation)...\n")

    # Το νέο "Χρυσό Όριο" βασισμένο στα δικά σου πειραματικά δεδομένα!
    threshold_dist = 1.0

    validator = PlanetValidator(input_length=201)
    validator.model = tf.keras.models.load_model('models/planet_validator_realistic.h5')
    planet_centroid = np.load('models/planet_centroid.npy', allow_pickle=True)
    prep = Preprocessor(normalization_type='zscore')

    targets_to_scan = [
        (37749396, 0.941),  # WASP-18 b (Confirmed Planet)
        (281541555, 1.4),  # Noise (False Positive)
        (261136679, 6.267),  # Pi Mensae c (Confirmed Planet)
        (158324245, 1.493)  # Eclipsing Binary
    ]

    results = []

    for tic_id, period in targets_to_scan:
        print(f"🔍 Ελέγχεται το TIC {tic_id} (Περίοδος: {period}d)...", end=" ")

        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            if len(search) == 0:
                print("❌ Δεν βρέθηκαν δεδομένα.")
                continue

            lc = search[0].download()
            if lc is None:
                print("❌ Κενό download.")
                continue

            time = np.array(lc.time.value, dtype=np.float64)
            flux = np.array(lc.flux.value, dtype=np.float64)

            mask = ~np.isnan(flux)
            time, flux = time[mask], flux[mask]

            time_clean, flux_norm, _, _ = prep.process(time, flux)
            phase, flux_folded = fold_lightcurve(time_clean, flux_norm, period)

            # --- DUAL-GATE INFERENCE ---
            prob = float(np.squeeze(validator.predict(phase, flux_folded)))
            features = validator.get_features(phase, flux_folded)
            dist = float(distance.euclidean(features, planet_centroid))

            # --- DECISION LOGIC ---
            status = "UNKNOWN"
            if prob > 0.5:
                if dist <= threshold_dist:
                    status = "✅ PLANET CANDIDATE"
                else:
                    status = "❌ ANOMALY (Noise/Artifact)"
            else:
                status = "🪨 NO PLANET"

            print(status)

            results.append({
                "TIC_ID": tic_id,
                "Period": period,
                "Probability": round(prob, 4),
                "Distance": round(dist, 4),
                "Verdict": status
            })

        except Exception as e:
            print(f"💥 Σφάλμα: {e}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs('reports', exist_ok=True)
        report_path = 'reports/scan_results.csv'
        df.to_csv(report_path, index=False)
        print("\n" + "=" * 50)
        print(f"📁 Η σάρωση ολοκληρώθηκε! Έτοιμο το: {report_path}")
        print("=" * 50)


if __name__ == "__main__":
    run_batch_scan()
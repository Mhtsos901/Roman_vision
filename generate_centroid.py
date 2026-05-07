import numpy as np
import tensorflow as tf
from src.models.planet_validator import PlanetValidator
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve
import lightkurve as lk
import os


def generate():
    print("🎯 Ξεκινάει η δημιουργία του Planet Centroid...")

    # 1. Setup
    validator = PlanetValidator(input_length=201)
    model_path = 'models/planet_validator_realistic.h5'

    if not os.path.exists(model_path):
        print(f"❌ Το μοντέλο δεν βρέθηκε στο {model_path}")
        return

    validator.model = tf.keras.models.load_model(model_path)

    # Ρητή δήλωση zscore για να είμαστε 100% ευθυγραμμισμένοι με το test_planet.py
    prep = Preprocessor(normalization_type='zscore')

    # 2. Λίστα με Confirmed Planets (εμπλουτισμένη για καλύτερο αντιπροσωπευτικό δείγμα)
    targets = [
        (25155310, 3.288),  # WASP-126 b
        (261136679, 6.267),  # Pi Mensae c
        (37749396, 0.941),  # WASP-18 b
        (118335442, 5.331),  # TOI-1601 b
        (307310363, 4.653)  # TOI-125 b
    ]

    all_embeddings = []

    for tic, per in targets:
        try:
            print(f"  → Επεξεργασία TIC {tic}...")
            search = lk.search_lightcurve(f"TIC {tic}", mission='TESS')

            if len(search) == 0:
                print(f"    ⚠️ Δεν βρέθηκαν δεδομένα για το TIC {tic}.")
                continue

            # Κατεβάζουμε το πρώτο διαθέσιμο (συνήθως SPOC)
            lc = search[0].download()
            if lc is None: continue

            # Preprocessing (Z-score) & Embedding Extraction
            # Χρησιμοποιούμε το .value για να έχουμε καθαρά numpy arrays
            time_clean, flux_norm, _, _ = prep.process(lc.time.value, lc.flux.value)
            phase, flux_folded = fold_lightcurve(time_clean, flux_norm, per)

            # Εξαγωγή των 64 χαρακτηριστικών (Embeddings)
            emb = validator.get_features(phase, flux_folded)
            all_embeddings.append(emb)
            print(f"    ✅ Επιτυχής εξαγωγή features.")

        except Exception as e:
            print(f"    ❌ Σφάλμα στο TIC {tic}: {e}")

    # 3. Υπολογισμός και Αποθήκευση
    if len(all_embeddings) > 0:
        # Υπολογίζουμε τον μέσο όρο όλων των διανυσμάτων
        centroid = np.mean(all_embeddings, axis=0)
        centroid = np.array(centroid, dtype=np.float32)

        os.makedirs('models', exist_ok=True)
        np.save('models/planet_centroid.npy', centroid)

        print("\n" + "=" * 40)
        print(f"✅ Επιτυχία! Το Centroid δημιουργήθηκε από {len(all_embeddings)} πλανήτες.")
        print(f"📏 Σχήμα Centroid: {centroid.shape}")
        print(f"📍 Αποθηκεύτηκε στο: models/planet_centroid.npy")
        print("=" * 40)
    else:
        print("\n❌ Αποτυχία: Δεν συλλέχθηκε κανένα embedding.")


if __name__ == "__main__":
    generate()
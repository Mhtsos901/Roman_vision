import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.planet_validator import PlanetValidator
from src.models.trainer import ModelTrainer
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

# 1. Εκπαίδευση (αν δεν υπάρχει ήδη αποθηκευμένο μοντέλο)
import os
model_path = 'models/planet_validator.h5'
if not os.path.exists(model_path):
    print("Εκπαίδευση νέου μοντέλου...")
    validator = PlanetValidator(input_length=201)
    trainer = ModelTrainer(validator, input_length=201, n_samples_per_class=1000)
    history = trainer.train(epochs=20)
    validator.model.save(model_path)
    print(f"Μοντέλο αποθηκεύτηκε στο {model_path}")
else:
    print("Φόρτωση υπάρχοντος μοντέλου...")
    validator = PlanetValidator(input_length=201)
    validator.model = tf.keras.models.load_model(model_path)

# 2. Προσπάθεια εύρεσης τοπικού αρχείου WASP-18 b
tic_id = "25155310"
local_files = glob.glob(f'data/raw/*{tic_id}*')
if local_files:
    print(f"Βρέθηκε τοπικό αρχείο: {local_files[0]}")
    import lightkurve as lk
    lc = lk.read(local_files[0])
    preprocessor = Preprocessor(outlier_sigma=5.0, savgol_window=21)
    time_raw = lc.time.value
    flux_raw = lc.flux.value
    time_clean, flux_clean, _, _ = preprocessor.process(time_raw, flux_raw)
    period = 0.941452
    phase, flux_folded = fold_lightcurve(time_clean, flux_clean, period)
    prob = validator.predict(phase, flux_folded)
    print(f"\n🔮 Πιθανότητα ύπαρξης πλανήτη (WASP-18 b): {prob:.4f}")
    if prob > 0.5:
        print("   👉 Το CNN προβλέπει: ΠΙΘΑΝΟΣ ΕΞΩΠΛΑΝΗΤΗΣ!")
    else:
        print("   👉 Το CNN προβλέπει: ΜΑΛΛΟΝ ΔΕΝ ΥΠΑΡΧΕΙ ΠΛΑΝΗΤΗΣ.")
else:
    print("Δεν βρέθηκε τοπικό αρχείο. Χρήση συνθετικής βουτιάς για επίδειξη.")
    phase_grid = np.linspace(0, 1, 201)
    flux = np.ones(201)
    transit_width = 0.04
    in_transit = np.abs(phase_grid - 0.5) < transit_width
    flux[in_transit] -= 0.01
    flux += np.random.normal(0, 0.0005, 201)
    flux = (flux - flux.min()) / (flux.max() - flux.min())
    input_data = flux.reshape(1, 201, 1)
    prob = validator.model.predict(input_data, verbose=0)[0][0]
    print(f"Πιθανότητα πλανήτη (συνθετική βουτιά): {prob:.4f}")
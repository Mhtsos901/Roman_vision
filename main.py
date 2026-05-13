import os
import csv
import time
import random
import logging
import threading
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import lightkurve as lk
import concurrent.futures
from scipy.spatial import distance
from astropy.timeseries import BoxLeastSquares

# === ΤΑ IMPORTS ΠΟΥ ΕΙΧΑΝ ΧΑΘΕΙ (Οι δικές σου κλάσεις!) ===
from src.models.planet_validator import PlanetValidator
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve
# ==========================================================

# 1. Απενεργοποίηση της μπάρας φόρτωσης της NASA
from astropy.utils.data import conf

conf.show_progress = False

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =======================================================================
# 📡 SETUP ΕΠΑΓΓΕΛΜΑΤΙΚΟΥ LOGGING (Τέλος τα print crashes!)
# =======================================================================
logger = logging.getLogger("RomanVision")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stream_handler)

# Σωπαίνουμε τα εσωτερικά μηνύματα του lightkurve & astropy
logging.getLogger('lightkurve').setLevel(logging.ERROR)
logging.getLogger('astropy').setLevel(logging.ERROR)

# =======================================================================
# 🚦 ΦΑΝΑΡΙΑ ΑΣΦΑΛΕΙΑΣ
# =======================================================================
csv_lock = threading.Lock()
ai_lock = threading.Lock()


# =======================================================================
# 🛡️ LEVEL 1: Αλγοριθμικό Φίλτρο για Διπλά Άστρα (EB Pre-screener)
# =======================================================================
def check_for_eclipsing_binary(time_arr, flux_arr, period):
    flux_norm = flux_arr / np.median(flux_arr)
    phase = (time_arr % period) / period
    bins = np.linspace(0, 1, 101)
    bin_medians = np.full(100, 1.0)
    for i in range(100):
        m = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(m): bin_medians[i] = np.median(flux_norm[m])

    phi_0 = bins[np.argmin(bin_medians)] + 0.005
    aligned_phase = (phase - phi_0 + 0.5) % 1.0

    primary_mask = (aligned_phase > 0.45) & (aligned_phase < 0.55)
    secondary_mask = (aligned_phase < 0.05) | (aligned_phase > 0.95)
    out_mask = (aligned_phase > 0.20) & (aligned_phase < 0.30)
    if not np.any(out_mask): out_mask = ~primary_mask

    shifted_time = time_arr - (phi_0 * period)
    transit_index = np.floor((shifted_time / period) + 0.5)

    odd_mask = primary_mask & (transit_index % 2 != 0)
    even_mask = primary_mask & (transit_index % 2 == 0)

    if not np.any(odd_mask) or not np.any(even_mask):
        return False, "Not enough transits"

    baseline = np.median(flux_norm[out_mask])
    noise = np.std(flux_norm[out_mask])

    odd_depth = max(baseline - np.percentile(flux_norm[odd_mask], 1), 1e-6)
    even_depth = max(baseline - np.percentile(flux_norm[even_mask], 1), 1e-6)

    ratio = min(odd_depth, even_depth) / max(odd_depth, even_depth)
    sec_depth = 0
    if np.any(secondary_mask):
        sec_depth = baseline - np.percentile(flux_norm[secondary_mask], 1)

    if ratio < 0.85: return True, f"Odd/Even Ratio: {ratio * 100:.1f}%"
    if (sec_depth > 0.10 * max(odd_depth, even_depth)) and (sec_depth > 3 * noise):
        return True, f"Secondary Eclipse Found"

    return False, "Clean Signal"


# =======================================================================
# 🎯 LEVEL 1.5: BLS Signal Refinement
# =======================================================================
def refine_period_with_bls(time_arr, flux_arr, hint_period):
    try:
        model = BoxLeastSquares(time_arr, flux_arr)
        period_grid = np.linspace(hint_period * 0.95, hint_period * 1.05, 500)
        duration_grid = np.linspace(0.02, 0.2, 10)

        results = model.power(period_grid, duration_grid)
        best_idx = np.argmax(results.power)
        exact_period = results.period[best_idx]

        if abs(exact_period - hint_period) < (hint_period * 0.1):
            return exact_period
        else:
            return hint_period

    except Exception as e:
        return hint_period


# =======================================================================
# 🚀 THE WORKER
# =======================================================================
def process_single_target(tic_id, hint_period, validator, prep, planet_centroid, threshold_dist, report_path):
    try:
        sleep_time = random.uniform(0.1, 2.0)
        time.sleep(sleep_time)

        logger.info(f"[DOWNLOAD] TIC {tic_id:<10} | Downloading data...")
        search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
        if len(search) == 0:
            return f"[ERROR]    TIC {tic_id:<10} | No data found."

        lc = search[0].download()
        if lc is None:
            return f"[ERROR]    TIC {tic_id:<10} | Empty download."

        logger.info(f"[PROCESS]  TIC {tic_id:<10} | BLS & Filtering...")
        time_arr = np.array(lc.time.value, dtype=np.float64)
        flux_arr = np.array(lc.flux.value, dtype=np.float64)
        mask = ~np.isnan(flux_arr)
        time_arr, flux_arr = time_arr[mask], flux_arr[mask]

        exact_period = refine_period_with_bls(time_arr, flux_arr, hint_period)
        is_eb, eb_reason = check_for_eclipsing_binary(time_arr, flux_arr, exact_period)

        if is_eb:
            status_csv = f"❌ ECLIPSING BINARY ({eb_reason})"
            status_console = f"ECLIPSING BINARY ({eb_reason})"
            prob, dist = 0.0, 0.0
        else:
            logger.info(f"[AI]       TIC {tic_id:<10} | Neural Network Prediction...")
            time_clean, flux_norm, _, _ = prep.process(time_arr, flux_arr)
            phase, flux_folded = fold_lightcurve(time_clean, flux_norm, exact_period)

            with ai_lock:
                prob = float(np.squeeze(validator.predict(phase, flux_folded)))
                features = validator.get_features(phase, flux_folded)

            dist = float(distance.euclidean(features, planet_centroid))

            if prob > 0.5:
                if dist <= threshold_dist:
                    status_csv = "✅ PLANET CANDIDATE"
                    status_console = "PLANET CANDIDATE"
                else:
                    status_csv = "❌ ANOMALY (Noise/Artifact)"
                    status_console = "ANOMALY (Noise/Artifact)"
            else:
                status_csv = "🪨 NO PLANET"
                status_console = "NO PLANET"

        with csv_lock:
            with open(report_path, mode='a', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow([
                    tic_id, exact_period, round(prob, 4) if not is_eb else "-", round(dist, 4) if not is_eb else "-",
                    status_csv
                ])

        return f"[SUCCESS]  TIC {tic_id:<10} | {status_console} (Dist: {dist:.3f})"

    except Exception as e:
        return f"[ERROR]    TIC {tic_id:<10} | Exception: {str(e)}"


# =======================================================================
# 🚀 ΚΥΡΙΟ PIPELINE (Multithreaded Orchestrator)
# =======================================================================
def run_batch_scan():
    logger.info("\n" + "=" * 70)
    logger.info("=== ROMAN VISION PIPELINE (PARALLEL MULTITHREADING) ===")
    logger.info("=" * 70 + "\n")

    threshold_dist = 1.0

    validator = PlanetValidator(input_length=201)
    validator.model = tf.keras.models.load_model(os.path.join('models', 'planet_validator_realistic.h5'))
    planet_centroid = np.load(os.path.join('models', 'planet_centroid.npy'), allow_pickle=True)
    prep = Preprocessor(normalization_type='zscore')

    csv_path = os.path.join('data', 'interim', 'targets_to_scan.csv')
    if not os.path.exists(csv_path):
        logger.info(f"[FATAL] Cannot find {csv_path}. Please run data_extractor.py.")
        return

    try:
        df_targets = pd.read_csv(csv_path, encoding='utf-8-sig')
        df_targets = df_targets.dropna(subset=['TIC_ID', 'Period'])
        targets_to_scan = list(zip(df_targets['TIC_ID'].astype(int), df_targets['Period']))
        logger.info(f"[INFO] Loaded {len(targets_to_scan)} targets.")
    except Exception as e:
        logger.info(f"[FATAL] Error reading CSV: {e}")
        return

    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'scan_results.csv')

    with open(report_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["TIC_ID", "Period", "Probability", "Distance", "Verdict"])

    logger.info(f"\n[SYSTEM] Starting 4 Worker Threads...\n")
    logger.info("-" * 70)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_single_target, tic, period, validator, prep, planet_centroid, threshold_dist,
                                report_path) for tic, period in targets_to_scan]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result_message = future.result()
                    logger.info(result_message)
                    logger.info("-" * 40)
                except Exception as e:
                    logger.info(f"[FATAL] Thread crashed: {e}")

    except KeyboardInterrupt:
        logger.info("\n" + "!" * 50)
        logger.info("[WARNING] USER INTERRUPT!")
        logger.info("[WARNING] Data safely saved to CSV.")
        logger.info("!" * 50 + "\n")

    logger.info("-" * 70)
    logger.info(f"[SYSTEM] Pipeline Completed! Report saved to: {report_path}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    run_batch_scan()
import pandas as pd
import numpy as np  # <-- Προστέθηκε για τον υπολογισμό του κέντρου
import lightkurve as lk
import matplotlib.pyplot as plt
import warnings

# Αγνόηση προειδοποιήσεων για πιο καθαρό terminal
warnings.filterwarnings('ignore')


def plot_candidate(tic_id):
    print(f"🎨 Ξεκινάει η οπτικοποίηση για το TIC {tic_id}...")

    # 1. Ανάγνωση του CSV για να βρούμε την Περίοδο (Period)
    try:
        df = pd.read_csv('reports/scan_results.csv', encoding='utf-8-sig')
        target_row = df[df['TIC_ID'] == tic_id]

        if target_row.empty:
            print(f"❌ Το TIC {tic_id} δεν υπάρχει στο report σου!")
            return

        verdict = target_row.iloc[0]['Verdict']
        period = target_row.iloc[0]['Period']

        print(f"📊 Στο Report καταγράφηκε ως: {verdict} (Περίοδος: {period} μέρες)")

    except Exception as e:
        print(f"❌ Σφάλμα ανάγνωσης του CSV: {e}")
        return

    # 2. Λήψη Δεδομένων (Download)
    print("⏳ Λήψη δεδομένων από τους servers της NASA...")
    search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
    if len(search) == 0:
        print("❌ Δεν βρέθηκαν δεδομένα για κατέβασμα.")
        return

    # Κατεβάζουμε και "καθαρίζουμε" το γράφημα (Outlier Removal & Detrending)
    lc = search[0].download().remove_nans().normalize().remove_outliers(sigma_upper=3, sigma_lower=10).flatten(
        window_length=101)

    # 3. Δημιουργία Γραφημάτων (Plotting)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.canvas.manager.set_window_title(f'Roman Vision - Target: {tic_id}')

    # --- TOP PANEL: Ολόκληρη η καμπύλη (Raw Lightcurve) ---
    lc.scatter(ax=ax1, color='black', alpha=0.5, s=5, label='Normalized Flux')
    ax1.set_title(f"Full Lightcurve - TIC {tic_id}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("Time (Days)")

    # --- BOTTOM PANEL: Διπλωμένη και Ζουμαρισμένη (Phase Folded) ---

    # ΒΗΜΑ Α: Προσωρινό δίπλωμα για να βρούμε πού "κάθεται" η βουτιά
    temp_fold = lc.fold(period=period)
    temp_binned = temp_fold.bin(time_bin_size=0.005)

    # ΒΗΜΑ Β: Υπολογισμός του Κέντρου Έκλειψης (Epoch)
    min_index = np.argmin(temp_binned.flux.value)
    min_phase = temp_binned.phase.value[min_index]
    epoch_time = lc.time.value[0] + (min_phase * period)

    # ΒΗΜΑ Γ: Τελικό, απόλυτα κεντραρισμένο δίπλωμα
    folded_lc = lc.fold(period=period, epoch_time=epoch_time)

    # Ζωγραφίζουμε τα αρχικά σημεία (γκρι)
    folded_lc.scatter(ax=ax2, color='gray', alpha=0.3, s=5, label='Folded Data')

    # Κάνουμε Binning (ομαδοποίηση)
    binned_lc = folded_lc.bin(time_bin_size=0.005)
    binned_lc.scatter(ax=ax2, color='red', s=40, edgecolor='black', label=f'Binned (Period: {period:.4f}d)', zorder=10)

    # Ζουμάρουμε με ασφάλεια στο 0.0, αφού ξέρουμε ότι η βουτιά είναι πλέον εκεί!
    ax2.set_xlim(-0.15, 0.15)
    ax2.set_title("Phase Folded Lightcurve (Transit Zoom)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Relative Flux")
    ax2.set_xlabel("Phase")

    # Βελτιώσεις εμφάνισης
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    target_tic = 169904935
    plot_candidate(target_tic)
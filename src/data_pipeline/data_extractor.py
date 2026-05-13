import os
import pandas as pd

input_file = os.path.join('data', 'raw', 'mastDownload', 'TOI_2026.05.13_08.44.36.csv')
output_file = os.path.join('data', 'interim', 'targets_to_scan.csv')

print(f"🔄 Ανάγνωση του αρχείου της NASA: {input_file}...")

try:
    # Διαβάζουμε το CSV αγνοώντας τα σχόλια/headers της NASA
    df = pd.read_csv(input_file, comment='#')

    # Πιθανές ονομασίες που χρησιμοποιεί η NASA
    tic_keywords = ['TIC', 'TID']
    period_keywords = ['PERIOD', 'ORBPER']

    # Ψάχνουμε με list comprehension. Αν δεν βρει τίποτα, επιστρέφει None.
    tic_col = next((col for col in df.columns if any(k in col.upper() for k in tic_keywords)), None)
    period_col = next((col for col in df.columns if any(k in col.upper() for k in period_keywords)), None)

    # --- DEBUGGING BLOCK ---
    if not tic_col or not period_col:
        print("\n❌ Αποτυχία εύρεσης στηλών!")
        print("Ορίστε οι πρώτες 20 στήλες που περιέχει το αρχείο της NASA (για να δούμε πώς τις ονόμασαν):")
        print(list(df.columns)[:20])
        raise ValueError("Δεν βρέθηκαν οι απαιτούμενες στήλες TIC και Period.")
    # -----------------------

    print(f"✅ Εντοπίστηκαν οι στήλες: Στόχος -> '{tic_col}', Περίοδος -> '{period_col}'")

    # Καθαρισμός
    df_clean = df[[tic_col, period_col]].copy()
    df_clean.columns = ['TIC_ID', 'Period']
    df_clean = df_clean.dropna()

    # Παίρνουμε τους πρώτους 100 στόχους
    df_sample = df_clean.head(100)
    df_sample.to_csv(output_file, index=False)

    print("\n" + "="*50)
    print(f"🎉 ΕΠΙΤΥΧΙΑ! Το Data Extraction ολοκληρώθηκε.")
    print(f"📁 Δημιουργήθηκε το {output_file} με {len(df_sample)} στόχους.")
    print("🚀 Μπορείς πλέον να τρέξεις το batch_scanner.py!")
    print("="*50)

except Exception as e:
    print(f"❌ Σφάλμα: {e}")
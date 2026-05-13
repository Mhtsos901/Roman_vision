import os
import math
import pandas as pd
import numpy as np
from astroquery.mast import Catalogs
import warnings

# Κρύβουμε κάποια ενοχλητικά warnings της astroquery
warnings.filterwarnings('ignore')


def get_stellar_parameters(tic_id):
    """Ρωτάει τη NASA για τα στοιχεία του άστρου (Μάζα, Ακτίνα, Θερμοκρασία)"""
    try:
        catalog_data = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
        if len(catalog_data) > 0:
            star = catalog_data[0]

            # Αν τα δεδομένα λείπουν από τη βάση, επιστρέφουμε None
            if np.ma.is_masked(star['mass']) or np.ma.is_masked(star['Teff']) or np.ma.is_masked(star['rad']):
                return None, None, None

            return float(star['mass']), float(star['rad']), float(star['Teff'])
    except Exception as e:
        print(f"⚠️ Σφάλμα σύνδεσης με NASA για TIC {tic_id}: {e}")
    return None, None, None


def analyze_candidates():
    print("\n" + "=" * 60)
    print("🌍 ΕΚΚΙΝΗΣΗ ASTRO-ENGINE (ΕΥΡΕΣΗ ΚΑΤΟΙΚΙΣΙΜΗΣ ΖΩΝΗΣ)")
    print("=" * 60 + "\n")

    input_csv = os.path.join('reports', 'scan_results.csv')
    output_csv = os.path.join('reports', 'habitable_planets.csv')

    if not os.path.exists(input_csv):
        print(f"❌ Δεν βρέθηκε το {input_csv}!")
        return

    # 1. Διαβάζουμε το CSV και κρατάμε ΜΟΝΟ τους υποψήφιους πλανήτες
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    candidates = df[df['Verdict'].str.contains('PLANET CANDIDATE', na=False)]

    if candidates.empty:
        print("❌ Δεν βρέθηκαν '✅ PLANET CANDIDATE' στο report σου.")
        return

    print(f"🔍 Βρέθηκαν {len(candidates)} Πιθανοί Πλανήτες! Ξεκινάει η ανάλυση...\n")

    results = []

    for index, row in candidates.iterrows():
        tic_id = int(row['TIC_ID'])
        period_days = float(row['Period'])

        print(f"📡 Λήψη αστρικών δεδομένων για TIC {tic_id}...")
        mass, radius, teff = get_stellar_parameters(tic_id)

        if mass is None or teff is None:
            print(f"   ⚠️ Η NASA δεν έχει πλήρη δεδομένα για αυτό το άστρο. Προσπέραση.\n")
            continue

        # --- 🧮 ΜΑΘΗΜΑΤΙΚΑ ΑΣΤΡΟΦΥΣΙΚΗΣ ---

        # 1. Υπολογισμός Απόστασης Πλανήτη (Semi-major axis σε AU) - 3ος Νόμος Kepler
        period_years = period_days / 365.25
        distance_au = (mass * (period_years ** 2)) ** (1 / 3)

        # 2. Υπολογισμός Λαμπρότητας Άστρου (Luminosity σε σχέση με τον Ήλιο)
        luminosity = (radius ** 2) * ((teff / 5778) ** 4)

        # 3. Υπολογισμός Κατοικίσιμης Ζώνης (Habitable Zone Limits σε AU)
        hz_inner = math.sqrt(luminosity / 1.1)
        hz_outer = math.sqrt(luminosity / 0.53)

        # 4. Αξιολόγηση Θερμοκρασίας
        # 4. Αξιολόγηση Θερμοκρασίας (Με προστασία από NaNs)
        if math.isnan(distance_au) or math.isnan(hz_inner) or math.isnan(hz_outer):
            status = "❓ ΑΓΝΩΣΤΟ (Λείπουν αστρικά δεδομένα)"
        elif distance_au < hz_inner:
            status = "🔥 ΠΟΛΥ ΚΑΥΤΟΣ (Lava World)"
        elif distance_au > hz_outer:
            status = "❄️ ΠΟΛΥ ΚΡΥΟΣ (Ice World)"
        else:
            status = "🌍 ΚΑΤΟΙΚΙΣΙΜΗ ΖΩΝΗ! (Liquid Water Possible)"

        print(f"   ⭐ Άστρο: Μάζα={mass:.2f} M☉ | Θερμοκρασία={teff:.0f}K")
        print(f"   🪐 Πλανήτης: Περίοδος={period_days:.2f} μέρες | Απόσταση={distance_au:.4f} AU")
        print(f"   🎯 Κατοικίσιμη Ζώνη Άστρου: Από {hz_inner:.3f} AU έως {hz_outer:.3f} AU")
        print(f"   ➡️ ΚΑΤΑΣΤΑΣΗ: {status}\n")

        results.append({
            'TIC_ID': tic_id,
            'Period_Days': round(period_days, 4),
            'Star_Mass': mass,
            'Star_Temp_K': teff,
            'Planet_Dist_AU': round(distance_au, 4),
            'HZ_Inner_AU': round(hz_inner, 4),
            'HZ_Outer_AU': round(hz_outer, 4),
            'Habitability': status
        })

    # Αποθήκευση στο νέο CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print("=" * 60)
        print(f"✅ Η ανάλυση ολοκληρώθηκε! Τα αποτελέσματα σώθηκαν στο: {output_csv}")
        print("=" * 60)


if __name__ == "__main__":
    analyze_candidates()
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import tensorflow as tf
import lightkurve as lk
import warnings

# Εισαγωγή των δικών σου modules
from src.models.planet_validator import PlanetValidator
from src.data_pipeline.preprocessor import Preprocessor
from src.models.folder import fold_lightcurve

# Απόκρυψη ενοχλητικών warnings
warnings.filterwarnings('ignore')


def get_project_paths():
    """
    Υπολογίζει δυναμικά τα μονοπάτια του project.
    """
    # Η διαδρομή αυτού του αρχείου
    current_file = os.path.abspath(__file__)
    # Ο φάκελος src/features/
    current_dir = os.path.dirname(current_file)
    # Η ρίζα του project (Roman_vision) - ανεβαίνουμε 2 επίπεδα
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    model_path = os.path.join(project_root, "models", "planet_validator_realistic.h5")
    return model_path


def fetch_embedding(validator, preprocessor, tic_id, period):
    """
    Κατεβάζει τα δεδομένα και εξάγει το 64-D feature vector (embedding).
    """
    try:
        print(f"  → Επεξεργασία TIC {tic_id}...")

        # 1. Ελαστική Αναζήτηση (Flexible Search)
        # Αφαιρούμε το author='SPOC' για να επιτρέψουμε στο Lightkurve να βρει
        # δεδομένα και από το QLP (Quick Look Pipeline), όπου βρίσκονται τα περισσότερα FP.
        search = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')

        if len(search) == 0:
            print(f"    ⚠️ Δεν βρέθηκαν δεδομένα για το TIC {tic_id}")
            return None

        # 2. Επιλογή του καλύτερου διαθέσιμου Light Curve
        # Αν υπάρχουν πολλά sectors, παίρνουμε το πρώτο.
        # Προσθέτουμε .download() με προσοχή.
        lc = search[0].download()

        if lc is None:
            print(f"    ⚠️ Αποτυχία λήψης (download) για το TIC {tic_id}")
            return None

        # 3. Μετατροπή σε Numpy Arrays και καθαρισμός
        # Χρησιμοποιούμε .value για να πάρουμε τα ωμά δεδομένα από τα Lightkurve objects
        time = lc.time.value
        flux = lc.flux.value

        # 4. Preprocessing & Folding (Data Pipeline)
        # Εδώ εφαρμόζουμε το normalization και το σπάσιμο σε Global/Local views
        time_clean, flux_clean, _, _ = preprocessor.process(time, flux)
        phase, flux_folded = fold_lightcurve(time_clean, flux_clean, period)

        # 5. Εξαγωγή Embeddings (Feature Extraction)
        # Καλούμε τη μέθοδο που διορθώσαμε στο PlanetValidator
        embedding = validator.get_features(phase, flux_folded)

        return embedding

    except Exception as e:
        # Detailed Error Logging για να ξέρουμε ακριβώς τι "έσπασε"
        print(f"    ❌ Σφάλμα στο TIC {tic_id}: {type(e).__name__} - {e}")
        return None


def main():
    model_file = get_project_paths()

    if not os.path.exists(model_file):
        print(f"❌ Το αρχείο μοντέλου δεν βρέθηκε στη διαδρομή: {model_file}")
        return

    # 1. Αρχικοποίηση Μοντέλου και Preprocessor
    print("🤖 Φόρτωση μοντέλου και προετοιμασία...")
    validator = PlanetValidator()
    validator.model = tf.keras.models.load_model(model_file)
    prep = Preprocessor()

    # 2. Ορισμός Στόχων για το Visualization (TIC, Period, Label)
    # Labels: 0 = Noise, 1 = Confirmed Planet, 2 = False Positive (EB)
    targets = [
        # --- CONFIRMED PLANETS (Blue) ---
        (25155310, 3.288, 1),  # WASP-126 b
        (118335442, 5.331, 1),  # TOI-1601 b
        (261136679, 6.267, 1),  # Pi Mensae c

        # --- NOISE / QUIET STARS (Gray) ---
        (142748283, 5.000, 0),
        (25063461, 2.500, 0),

        # --- FALSE POSITIVES / ECLIPSING BINARIES (Red) ---
        (158324245, 1.493, 2),  # Το "99%" False Positive σου
        (150353011, 3.334, 2),  # Eclipsing Binary
        (270144888, 2.083, 2)  # Eclipsing Binary
    ]

    all_embeddings = []
    all_labels = []
    successful_tics = []

    # 3. Συλλογή Embeddings
    for tic, per, lab in targets:
        emb = fetch_embedding(validator, prep, tic, per)
        if emb is not None:
            all_embeddings.append(emb)
            all_labels.append(lab)
            successful_tics.append(tic)

    if not all_embeddings:
        print("❌ Δεν συλλέχθηκαν δεδομένα.")
        return

    data = np.array(all_embeddings)

    # 4. UMAP Dimensionality Reduction (64D -> 2D)
    print("\n📉 Εκτέλεση UMAP (Dimensionality Reduction)...")
    # Χρησιμοποιούμε n_neighbors=5 γιατί έχουμε λίγα δείγματα τώρα
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', random_state=42)
    embedding_2d = reducer.fit_transform(data)

    # 5. Οπτικοποίηση (Visualization)
    print("🎨 Δημιουργία γραφήματος...")
    plt.figure(figsize=(12, 8))

    colors = ['#808080', '#1f77b4', '#d62728']  # Gray, Blue, Red
    names = ['Noise/Quiet', 'Confirmed Planet', 'False Positive (Anomaly)']

    for i in range(3):
        idx = [j for j, val in enumerate(all_labels) if val == i]
        if idx:
            plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1],
                        c=colors[i], label=names[i], s=150, edgecolors='black', alpha=0.8)

    plt.title("Latent Space Visualization (UMAP)\nΠώς το CNN αντιλαμβάνεται τα διαφορετικά σήματα", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Προσθήκη TIC IDs πάνω από τα σημεία για αναγνώριση
    # Προσθήκη TIC IDs πάνω από τα σημεία χρησιμοποιώντας τη λίστα των επιτυχημένων
    for i in range(len(embedding_2d)):
        plt.annotate(
            f"TIC {successful_tics[i]}",
            (embedding_2d[i, 0], embedding_2d[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8
        )

    plt.show()


if __name__ == "__main__":
    main()
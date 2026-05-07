import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d


class ExoMinerExtractor:
    """
    Εξάγει ένα διάνυσμα χαρακτηριστικών (embedding) από μια καμπύλη φωτός
    χρησιμοποιώντας ένα τροποποιημένο CNN τύπου ExoMiner++.
    """

    def __init__(self, weights_path=None, input_length=201, embedding_size=64):
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.model = self._build_model()

        if weights_path:
            try:
                full_model = tf.keras.models.load_model(weights_path)
                # Αντιγράφουμε βάρη μεταξύ κοινών επιπέδων
                for layer in self.model.layers:
                    if layer.name in [l.name for l in full_model.layers]:
                        try:
                            self.model.get_layer(layer.name).set_weights(
                                full_model.get_layer(layer.name).get_weights()
                            )
                        except:
                            pass
                print(f"Φόρτωση βαρών από: {weights_path}")
            except Exception as e:
                print(f"Δεν βρέθηκαν προ-εκπαιδευμένα βάρη. Χρήση τυχαίου δικτύου.\nΣφάλμα: {e}")

    def _build_model(self):
        """Φτιάχνει ένα απλό 1D CNN για την εξαγωγή χαρακτηριστικών."""
        input_layer = tf.keras.layers.Input(shape=(self.input_length, 1), name='flux_input')

        x = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same', name='conv1')(input_layer)
        x = tf.keras.layers.MaxPooling1D(2, name='pool1')(x)
        x = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same', name='conv2')(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(x)
        embedding = tf.keras.layers.Dense(self.embedding_size, activation=None, name='embedding')(x)

        return tf.keras.Model(inputs=input_layer, outputs=embedding, name='ExoMiner_FeatureExtractor')

    def preprocess_for_cnn(self, time, flux):
        """Προετοιμάζει την καμπύλη φωτός για το CNN."""
        # Resampling σε ομοιόμορφο πλέγμα
        f_interp = interp1d(time, flux, kind='linear',
                            bounds_error=False, fill_value=0.0)
        time_grid = np.linspace(time.min(), time.max(), self.input_length)
        flux_resampled = f_interp(time_grid)

        # Min-Max scaling
        flux_min = np.min(flux_resampled)
        flux_max = np.max(flux_resampled)
        if flux_max - flux_min > 1e-9:
            flux_scaled = (flux_resampled - flux_min) / (flux_max - flux_min)
        else:
            flux_scaled = flux_resampled

        return flux_scaled.reshape(1, self.input_length, 1)

    def extract_features(self, time, flux):
        """Κύρια μέθοδος: επιστρέφει το embedding ως numpy array."""
        cnn_input = self.preprocess_for_cnn(time, flux)
        features = self.model.predict(cnn_input, verbose=0)
        return features.flatten()
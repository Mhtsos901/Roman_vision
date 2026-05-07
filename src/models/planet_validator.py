import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d


class PlanetValidator:
    def __init__(self, input_length=201):
        self.input_length = input_length
        self.model = self._build_model()

    def _build_model(self):
        # --- 1. Global View Branch ---
        global_input = tf.keras.layers.Input(shape=(self.input_length, 1), name='global_input')
        xg = tf.keras.layers.Conv1D(16, 11, activation='relu', padding='same')(global_input)
        xg = tf.keras.layers.BatchNormalization()(xg)
        xg = tf.keras.layers.MaxPooling1D(2)(xg)
        xg = tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same')(xg)
        xg = tf.keras.layers.BatchNormalization()(xg)
        xg = tf.keras.layers.GlobalAveragePooling1D()(xg)

        # --- 2. Local View Branch ---
        local_input = tf.keras.layers.Input(shape=(self.input_length, 1), name='local_input')
        xl = tf.keras.layers.Conv1D(16, 11, activation='relu', padding='same')(local_input)
        xl = tf.keras.layers.BatchNormalization()(xl)
        xl = tf.keras.layers.MaxPooling1D(2)(xl)
        xl = tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same')(xl)
        xl = tf.keras.layers.BatchNormalization()(xl)
        xl = tf.keras.layers.GlobalAveragePooling1D()(xl)

        # --- 3. Merging the Branches ---
        merged = tf.keras.layers.Concatenate()([xg, xl])

        # --- 4. Fully Connected Layers ---
        dense = tf.keras.layers.Dense(64, activation='relu', name='feature_layer')(merged)
        dense = tf.keras.layers.Dropout(0.4)(dense)
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense)

        model = tf.keras.Model(inputs=[global_input, local_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess(self, phase, flux):
        # 1. Global View (Όλη η φάση [0, 1])
        sort_idx = np.argsort(phase)
        f_global = interp1d(phase[sort_idx], flux[sort_idx], kind='linear', bounds_error=False, fill_value=0.0)
        global_grid = np.linspace(0, 1, self.input_length)
        flux_global = f_global(global_grid)

        mean_g, std_g = np.mean(flux_global), np.std(flux_global)
        flux_global_scaled = (flux_global - mean_g) / std_g if std_g > 1e-9 else flux_global - mean_g

        # 2. Local View (Ζουμ γύρω από τη διάβαση. Έστω phase [0.4, 0.6])
        local_grid = np.linspace(0.4, 0.6, self.input_length)
        flux_local = f_global(local_grid)  # Προσοχή: Χρησιμοποιούμε ξανά την ίδια interpolation function!

        mean_l, std_l = np.mean(flux_local), np.std(flux_local)
        flux_local_scaled = (flux_local - mean_l) / std_l if std_l > 1e-9 else flux_local - mean_l

        return [flux_global_scaled.reshape(1, self.input_length, 1),
                flux_local_scaled.reshape(1, self.input_length, 1)]

    def predict(self, phase, flux):
        # Τώρα το input_data είναι μια λίστα με 2 στοιχεία: [global_input, local_input]
        input_data = self.preprocess(phase, flux)
        probability = self.model.predict(input_data, verbose=0)[0][0]
        return probability

    def get_features(self, phase, flux):
        input_data = self.preprocess(phase, flux)

        feature_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('feature_layer').output
        )

        embeddings = feature_model.predict(input_data, verbose=0)
        return embeddings.flatten()
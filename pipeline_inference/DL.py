"""
Modulo per l'elaborazione dei dati utilizzando l'algoritmo K-Nearest Neighbors.

Questo modulo fornisce funzionalità per trovare i profili più simili
basati su vari tratti utilizzando l'algoritmo DL.
"""

import logging
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pipeline_inference.pipeline_inference_base import InferencePipelineBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model
import shap

logger = logging.getLogger(__name__)

class DLProcessor(InferencePipelineBase):
    """
    Processor per trovare i vicini più prossimi utilizzando l'algoritmo DL.
    """
    
    def __init__(self, model_path=None):
        """
        Inizializza il processor DL.
        
        Args:
            model_path (str, optional): Percorso del modello salvato da caricare.
        """
        super().__init__(name="DLProcessor")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.scaler = StandardScaler()
            self.model = Sequential([
                Dense(32, activation='relu', input_shape=(5,)),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')  
            ])
            logger.info(f"Inizializzato processor DL")
    
    def save_model(self, path=None):
        """
        Salva il modello DL su disco.
        
        Args:
            path (str, optional): Percorso dove salvare il modello. Se None, usa un nome predefinito.
            
        Returns:
            str: Percorso dove è stato salvato il modello.
            
        Raises:
            RuntimeError: Se il modello non è stato addestrato.
        """
        if self.model is None:
            raise RuntimeError("Nessun modello da salvare. Addestrare prima il modello.")
        
        try:
            # Directory del file corrente
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Salgo di un livello e punto alla cartella 'models'
            models_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "models"))
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info(f"Creata directory {models_dir}")

            # Salva il modello Keras
            keras_path = os.path.join(models_dir, "DL_model.h5")
            self.model.save(keras_path)

            # Salva lo scaler
            scaler_path = os.path.join(models_dir, "DL_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)

            logger.info(f"Modello DL salvato in: {keras_path}")
            logger.info(f"Scaler salvato in: {scaler_path}")
            return path
        except Exception as e:
            logger.error(f"Errore durante il salvataggio del modello: {e}")
            raise
    
    def load_model(self, path):
        """
        Carica un modello DL precedentemente salvato.
        
        Args:
            path (str): Percorso del modello da caricare.
            
        Raises:
            FileNotFoundError: Se il file del modello non esiste.
        """
        try:     
            keras_path = os.path.join(path, "DL_model.h5")
            scaler_path = os.path.join(path, "DL_scaler.pkl")     
            self.model = keras_load_model(keras_path)
            self.scaler = joblib.load(scaler_path)

            self.model = joblib.load(path)
            logger.info(f"Modello DL caricato da: {path}")
        except FileNotFoundError:
            logger.error(f"File del modello non trovato: {path}")
            raise
        except Exception as e:
            logger.error(f"Errore durante il caricamento del modello: {e}")
            raise
    
    def fit(self, records):
        """
        Addestra il modello DL sui dati forniti.
        
        Args:
            records (list): Lista di record con tratti di personalità.
            
        Returns:
            self: Il processor stesso.
            
        Raises:
            ValueError: Se i record non hanno il formato atteso.
        """
        try:
            self.records = records
            personality_vectors = []
            criticality_vector = []

            for record in records:
                if 'personality_traits' not in record:
                    raise ValueError("I record devono contenere personality_traits")
                
                traits = record['personality_traits']
                vector = [
                    traits.get('extraversion', 0),
                    traits.get('agreeableness', 0),
                    traits.get('conscientiousness', 0),
                    traits.get('neuroticism', 0),
                    traits.get('openness', 0)
                ]
                criticality = record['criticality_index']
                personality_vectors.append(vector)
                criticality_vector.append(criticality)
            
            self.data = np.array(personality_vectors)
            X_scaled = self.scaler.fit_transform(self.data)
            y = np.clip(np.array(criticality_vector), 0, 1)

            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            self.model.fit(X_scaled, y, epochs=50, batch_size=8, verbose=0)
            self.save_model()
            logger.info(f"Modello DL addestrato su {len(records)} record")
            return self
            
        except Exception as e:
            logger.error(f"Errore durante l'addestramento del modello DL: {e}")
            raise
    
    def process(self, traits):
        """
        Predice l'indice di criticità di un utente in base ai suoi tratti di personalità (OCEAN).
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            
        Returns:
            float: Indice di criticità stimato (da 0 a 1, approssimato a due decimali).
            
        Raises:
            ValueError: Se i dati di input non hanno il formato corretto
        """
        if isinstance(traits, dict):
            logger.info(f"Avvio procedura predizione criticità mediante DL")
        else:
            raise ValueError("Formato tratti di personalità non valido")
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modello o scaler non inizializzati. Caricare o addestrare prima il modello.")
        
        try:
            x_input = np.array([[
                traits.get('extraversion', 0),
                traits.get('agreeableness', 0),
                traits.get('conscientiousness', 0),
                traits.get('neuroticism', 0),
                traits.get('openness', 0)
            ]])

            x_scaled = self.scaler.transform(x_input)
            prediction = self.model.predict(x_scaled)[0][0]

            logger.info(f"Processo KNN completato. Indice di criticità stimato: {prediction:.4f}")


            explanation = self.explain(traits)
            logger.info(f"Baseline: {explanation['expected_value']:.4f}, Predizione: {explanation['predicted_value']:.4f}")
            for name, val in explanation['shap_values'].items():
                logger.info(f"{name:20s}: {val:+.4f}")

            return round(float(prediction), 2)
        except Exception as e:
            logger.error(f"Errore durante il processo DL: {e}")
            raise


    def explain(self, traits):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modello o scaler non inizializzati. Caricare o addestrare prima.")

        feature_names = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        x_input = np.array([[traits.get(f, 0) for f in feature_names]])
        x_scaled = self.scaler.transform(x_input)

        # Gestione dati di background
        if hasattr(self, 'data') and self.data is not None:
            num_background_samples = min(50, self.data.shape[0])
            background_unscaled = self.data[:num_background_samples]
            background_scaled = self.scaler.transform(background_unscaled)
        else:
            # Fallback se self.data non è disponibile (es. modello caricato senza fit)
            logger.warning("Dati di addestramento non trovati per il background SHAP. Uso l'input stesso come background.")
            background_scaled = x_scaled

        # Calcolo SHAP
        explainer = shap.KernelExplainer(self.model.predict, background_scaled)
        shap_values = explainer.shap_values(x_scaled) # Calcola valori SHAP per l'input

        # Estrai l'array numpy, gestendo il caso in cui SHAP restituisca una lista
        shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Calcolo Expected Value (robusto)
        raw_expected_value = explainer.expected_value
        if isinstance(raw_expected_value, (list, np.ndarray)):
            expected_value = np.ravel(raw_expected_value)[0]
        else:
            expected_value = raw_expected_value
        expected_value = float(expected_value) # Assicura sia float scalare

        # --- FIX Chiave: Squeeze SHAP values a 1D ---
        # Assicurati sia un array numpy
        if not isinstance(shap_vals_raw, np.ndarray):
            shap_vals_raw = np.array(shap_vals_raw)

        # Usa np.squeeze() per rimuovere dimensioni superflue (es. da (5, 1) a (5,))
        shap_values_1d = np.squeeze(shap_vals_raw)

        # Verifica che il risultato sia effettivamente 1D
        if shap_values_1d.ndim != 1:
            error_msg = f"Fallimento nell'ottenere array SHAP 1D. Shape dopo squeeze: {shap_values_1d.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # print(f"DEBUG: Shape finale 1D: {shap_values_1d.shape}") # Debug opzionale

        # Calcola il valore predetto usando l'array 1D corretto
        predicted_value = expected_value + np.sum(shap_values_1d)

        # --- Chiamata al Plot (con array 1D) ---
        try:
            # Passa l'array shap_values_1d che ora ha la forma corretta (es. (5,))
            shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_values_1d,  # <--- USA L'ARRAY 1D CORRETTO
                feature_names=feature_names
            )
        except Exception as plot_error:
            # È buona pratica gestire errori specifici del plot,
            # specialmente se eseguito in ambienti senza GUI
            logger.warning(f"Impossibile generare il plot SHAP waterfall: {plot_error}")
            # Decidi se l'errore di plot è bloccante o meno
            # raise plot_error # Decommenta se vuoi che l'errore fermi l'esecuzione

        # --- Return Dictionary (con array 1D) ---
        # Usa lo stesso array shap_values_1d per creare il dizionario
        return {
            "expected_value": expected_value,
            "predicted_value": predicted_value,
            "shap_values": dict(zip(feature_names, shap_values_1d)) # <--- USA L'ARRAY 1D CORRETTO
        }

    # def explain(self, traits):
    #     if self.model is None or self.scaler is None:
    #         raise RuntimeError("Modello o scaler non inizializzati. Caricare o addestrare prima.")
    #     feature_names = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    #     x_input = np.array([[traits.get(f, 0) for f in feature_names]])
    #     x_scaled = self.scaler.transform(x_input)
    #     if hasattr(self, 'data') and self.data is not None:
    #         background_scaled = self.scaler.transform(self.data[:50])
    #     else:
    #         background_scaled = x_scaled
    #     explainer = shap.KernelExplainer(self.model.predict, background_scaled)
    #     shap_values = explainer.shap_values(x_scaled)
    #     shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    #     #expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    #     # Estrazione robusta di expected_value per assicurarsi che sia uno scalare float
    #     raw_expected_value = explainer.expected_value
    #     if isinstance(raw_expected_value, (list, np.ndarray)):
    #         # Se è lista o array, appiattisci e prendi il primo elemento
    #         expected_value = np.ravel(raw_expected_value)[0]
    #     else:
    #         # Altrimenti, assumi sia già uno scalare
    #         expected_value = raw_expected_value
    #     # Converti esplicitamente a float Python nativo
    #     expected_value = float(expected_value)


    #     # Estrai i valori SHAP per la singola istanza (prima riga di shap_vals)
    #     # Assumendo che shap_vals sia almeno 1D
    #     shap_values_single_instance = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    #     # Calcola il valore predetto dalla somma di baseline e SHAP values
    #     # Usiamo l'array 1D dei valori SHAP per la singola istanza
    #     predicted_value = expected_value + np.sum(shap_values_single_instance)

    #     # Estrai i valori SHAP per la singola istanza come array 1D
    #     # Questo passaggio assicura che shap_values_single sia 1D
    #     shap_values_single = shap_vals[0] if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2 else shap_vals
    #     # Ulteriore sicurezza se shap_vals fosse > 2D o strano
    #     if shap_values_single.ndim > 1:
    #         shap_values_single = shap_values_single[0]

    #     # DEBUG: Controlla la forma prima del plot
    #     print(f"DEBUG: Shape di shap_values_single prima del plot: {shap_values_single.shape}")
    #     print(f"DEBUG: Tipo di shap_values_single prima del plot: {type(shap_values_single)}")

    #     # Chiamata originale alla funzione di plotting (può ancora dare problemi
    #     # in ambienti non grafici, ma l'errore 'truth value' da expected_value
    #     # dovrebbe essere risolto)
    #     shap.plots._waterfall.waterfall_legacy(
    #         expected_value,           # Ora è sicuramente uno скаляр float
    #         shap_values_single,       # Dovrebbe essere l'array 1D dei valori SHAP
    #         feature_names=feature_names
    #     )

    #     # Ritorna i risultati assicurandoti che i valori SHAP siano associati correttamente
    #     # Usa shap_values_single che è l'array 1D garantito
    #     return {
    #         "expected_value": expected_value,
    #         "predicted_value": predicted_value,
    #         "shap_values": dict(zip(feature_names, shap_values_single))
    #     }
    #     # predicted_value = expected_value + np.sum(shap_vals[0])
    #     # # Mostra solo la spiegazione per la prima istanza
    #     # shap_values_single = shap_vals[0] if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2 else shap_vals
    #     # shap.plots._waterfall.waterfall_legacy(
    #     #     expected_value,
    #     #     shap_values_single,
    #     #     feature_names=feature_names
    #     # )
    #     # return {
    #     #     "expected_value": expected_value,
    #     #     "predicted_value": predicted_value,
    #     #     "shap_values": dict(zip(feature_names, shap_vals[0]))
    #     # }
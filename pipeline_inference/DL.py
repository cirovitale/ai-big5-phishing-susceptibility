"""
Modulo per l'elaborazione dei dati utilizzando l'algoritmo di Deep Learning.

Questo modulo fornisce funzionalità per applicare un modello di rete neurale sull'argomento trattato.
"""

import logging
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from pipeline_inference.pipeline_inference_base import InferencePipelineBase
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense
from keras.optimizers import Adam
import shap

logger = logging.getLogger(__name__)

class DLProcessor(InferencePipelineBase):
    """
    Processor per predirre la suscettibilità dell'utente con tecniche di DL.
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
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "models"))
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info(f"Creata directory {models_dir}")

            keras_path = os.path.join(models_dir, "DL_model.h5")
            self.model.save(keras_path)

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

            logger.info(f"Processo DL completato. Indice di criticità stimato: {prediction:.4f}")


            explanation = self.explain(traits)
            logger.info(f"Baseline: {explanation['expected_value']:.4f}, Predizione: {explanation['predicted_value']:.4f}")
            for name, val in explanation['shap_values'].items():
                logger.info(f"{name:20s}: {val:+.4f}")

            return round(float(prediction), 2)
        except Exception as e:
            logger.error(f"Errore durante il processo DL: {e}")
            raise


    def explain(self, traits):
        """
        Effettua l'explanation del modello di DL mediante SHAP
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            
        Returns:
            dict: Valori ottenuti mediante SHAP
            
        Raises:
            RuntimeError: Se i modelli non sono stati inizializzati
        """

        if self.model is None or self.scaler is None:
            raise RuntimeError("Modello o scaler non inizializzati. Caricare o addestrare prima.")

        feature_names = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        x_input = np.array([[traits.get(f, 0) for f in feature_names]])
        x_scaled = self.scaler.transform(x_input)

        if hasattr(self, 'data') and self.data is not None:
            num_background_samples = min(50, self.data.shape[0])
            background_unscaled = self.data[:num_background_samples]
            background_scaled = self.scaler.transform(background_unscaled)
        else:
            logger.warning("Dati di addestramento non trovati per il background SHAP. Uso l'input stesso come background.")
            background_scaled = x_scaled

        explainer = shap.KernelExplainer(self.model.predict, background_scaled)
        shap_values = explainer.shap_values(x_scaled)

        shap_vals_raw = shap_values[0] if isinstance(shap_values, list) else shap_values

        raw_expected_value = explainer.expected_value
        if isinstance(raw_expected_value, (list, np.ndarray)):
            expected_value = np.ravel(raw_expected_value)[0]
        else:
            expected_value = raw_expected_value
        expected_value = float(expected_value)

        if not isinstance(shap_vals_raw, np.ndarray):
            shap_vals_raw = np.array(shap_vals_raw)

        shap_values_1d = np.squeeze(shap_vals_raw)

        if shap_values_1d.ndim != 1:
            error_msg = f"Fallimento nell'ottenere array SHAP 1D. Shape dopo squeeze: {shap_values_1d.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        predicted_value = expected_value + np.sum(shap_values_1d)

        try:
            shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_values_1d,
                feature_names=feature_names
            )
        except Exception as plot_error:
            logger.warning(f"Impossibile generare il plot SHAP waterfall: {plot_error}")


        return {
            "expected_value": expected_value,
            "predicted_value": predicted_value,
            "shap_values": dict(zip(feature_names, shap_values_1d)) 
        }

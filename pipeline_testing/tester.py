"""
Modulo per il testing e la valutazione delle predizioni con supporto per SHAP.
"""

import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione del logger
logger = logging.getLogger(__name__)

class Tester():
    """
    Classe per il testing e la valutazione delle performance dei modelli con SHAP.
    """
    
    def __init__(self, name="ModelTester"):
        """
        Inizializza il tester.
        
        Args:
            name (str): Nome del tester.
        """
        self.threshold = float(os.getenv('CRITICALITY_THRESHOLD', 0.6))
    
    def evaluate_predictions(self, predictions, true_values):
        """
        Valuta le predizioni confrontandole con i valori reali.
        
        Args:
            predictions (list): Lista di valori predetti (continui)
            true_values (list): Lista di valori reali (continui)
            
        Returns:
            dict: Dizionario con le metriche di valutazione
        """
        # Converti i valori continui in binari
        y_pred_binary = [1 if pred >= self.threshold else 0 for pred in predictions]
        y_true_binary = [1 if true >= self.threshold else 0 for true in true_values]
        
        # Calcola le metriche
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_binary, y_pred_binary).tolist(),
        }
        
        # Log dettagliato di tutte le metriche
        logger.info(f"Valutazione completata. Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    

    def process(self, data):
        """
        Esegue il testing sui dati forniti.

        Args:
            data (dict): Dizionario contenente 'predictions', 'true_values'
            
        Returns:
            dict: Metriche di valutazione
        """
        if not isinstance(data, dict) or 'predictions' not in data or 'true_values' not in data:
            raise ValueError("I dati devono essere un dizionario con 'predictions' e 'true_values'")
        
        predictions = data['predictions']
        true_values = data['true_values']
        
        
        # Valuta le predizioni
        metrics = self.evaluate_predictions(predictions, true_values)
        
        return metrics
"""
Modulo per l'ensemble di predittori per la valutazione della suscettibilità al phishing.

Questo modulo implementa un modello di ensemble che combina i risultati di diversi predittori
(KNN, Regressione, LLM, DL) utilizzando una media pesata per ottenere una predizione finale.
"""

import logging
from pipeline_inference.pipeline_inference_base import InferencePipelineBase
from config import ENSEMBLE_WEIGHT_KNN, ENSEMBLE_WEIGHT_REGRESSION, ENSEMBLE_WEIGHT_LLM, ENSEMBLE_WEIGHT_DL

logger = logging.getLogger(__name__)

class EnsembleProcessor(InferencePipelineBase):
    """
    Processor che combina i risultati di diversi predittori usando una media pesata.
    
    Formula: FinalPrediction = w0*KNN + w1*Regression + w2*LLM + w3*DL
    dove w0,...,w4 sono i pesi assegnati a ciascun predittore.
    """
    
    def __init__(self, weights=None, name="EnsembleProcessor"):
        """
        Inizializza il processor di ensemble.
        
        Args:
            weights (dict, optional): Dizionario con i pesi {predittore: peso}.
                                    Se None, verranno utilizzati pesi dalla configurazione.
            name (str): Nome del processor.
        """
        super().__init__(name=name)
        
        # Usa i pesi dalla configurazione se non specificati
        if weights is None:
            self.weights = {
                'KNN': ENSEMBLE_WEIGHT_KNN,
                'Regression': ENSEMBLE_WEIGHT_REGRESSION,
                'LLM': ENSEMBLE_WEIGHT_LLM,
                'DL': ENSEMBLE_WEIGHT_DL
            }
        else:
            self.weights = weights
            
        logger.info(f"Inizializzato {name} con pesi: {self.weights}")
    
    def calculate_ensemble(self, predictions):
        """
        Calcola il valore finale combinando i risultati dei diversi predittori.
        
        Args:
            predictions (dict): Dizionario con i risultati dei predittori.
            
        Returns:
            float: Valore finale calcolato.
            
        Raises:
            ValueError: Se le predizioni non hanno il formato atteso.
        """
        if not isinstance(predictions, dict):
            raise ValueError("Le predizioni devono essere fornite come dizionario")
        
        try:
            knn_value = float(predictions.get('KNN', 0))
            regression_value = float(predictions.get('Regression', 0))
            llm_value = float(predictions.get('LLM', 0))
            dl_value = float(predictions.get('DL', 0))
            
            # Calcolo pesato
            final_value = (
                self.weights['KNN'] * knn_value +
                self.weights['Regression'] * regression_value +
                self.weights['LLM'] * llm_value +
                self.weights['DL'] * dl_value
            )
            
            final_value = round(final_value, 4)
            
            logger.debug(f"Calcolato valore finale: {final_value} per predizioni: {predictions} con pesi: {self.weights}")
            return final_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Errore nella conversione dei valori: {e}")
            raise ValueError(f"Errore nel calcolo dell'ensemble: {e}")
    
    def process(self, predictions):
        """
        Calcola il valore finale combinando i risultati dei diversi predittori.
        
        Args:
            predictions: Risultati dei predittori come dizionario.
            
        Returns:
            float: Valore finale calcolato.
        """
        if isinstance(predictions, dict):
            return self.calculate_ensemble(predictions)
        else:
            raise ValueError("Formato tratti di personalità non valido")
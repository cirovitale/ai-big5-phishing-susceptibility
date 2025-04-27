"""
Modulo per l'ensemble di predittori per la valutazione della suscettibilità al phishing.

Questo modulo implementa un modello di ensemble che combina i risultati di diversi predittori
(KNN, Regressione, LLM, DL, LSTM) utilizzando una media pesata per ottenere una predizione finale.
"""

import logging
from pipeline_inference.pipeline_inference_base import InferencePipelineBase

logger = logging.getLogger(__name__)

class EnsembleProcessor(InferencePipelineBase):
    """
    Processor che combina i risultati di diversi predittori usando una media pesata.
    
    Formula: FinalPrediction = w0*KNN + w1*Regression + w2*LLM + w3*DL + w4*LSTM
    dove w0,...,w4 sono i pesi assegnati a ciascun predittore.
    """
    
    def __init__(self, weights=None, name="EnsembleProcessor"):
        """
        Inizializza il processor di ensemble.
        
        Args:
            weights (dict, optional): Dizionario con i pesi {predittore: peso}.
                                    Se None, verranno utilizzati pesi predefiniti.
            name (str): Nome del processor.
        """
        super().__init__(name=name)
        
        # Pesi predefiniti se non specificati
        if weights is None:
            self.weights = {
                'KNN': 1,               # w0
                'Regression': 1,        # w1
                'LLM': 0,               # w2
                'DL': 0,                # w3
                'LSTM': 0               # w4
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
            lstm_value = float(predictions.get('LSTM', 0))
            
            final_value = (
                self.weights['KNN'] * knn_value +
                self.weights['Regression'] * regression_value +
                self.weights['LLM'] * llm_value +
                self.weights['DL'] * dl_value +
                self.weights['LSTM'] * lstm_value
            )
            
            # final_value = round(final_value, 2) / len(final_value)
            final_value = round(final_value, 2) / 2 # TODO: cambiare, ora 2 perchè solo KNN e Regression
            
            logger.debug(f"Calcolato valore finale: {final_value} per predizioni: {predictions}")
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
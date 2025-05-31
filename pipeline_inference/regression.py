"""
Modulo per il calcolo del comportamento utente utilizzando una formula
di regressione lineare basata sui tratti di personalità.

Questo modulo implementa un modello di regressione lineare che predice
il comportamento dell'utente basato sui suoi tratti OCEAN.
"""

import logging
from pipeline_inference.pipeline_inference_base import InferencePipelineBase
from config import (
    REGRESSION_WEIGHT_OPENNESS,
    REGRESSION_WEIGHT_CONSCIENTIOUSNESS,
    REGRESSION_WEIGHT_EXTRAVERSION,
    REGRESSION_WEIGHT_AGREEABLENESS,
    REGRESSION_WEIGHT_NEUROTICISM
)

logger = logging.getLogger(__name__)

class RegressionProcessor(InferencePipelineBase):
    """
    Processor che calcola il comportamento utente usando una formula di regressione.
    
    Formula: UserBehavior = w0*O + w1*C + w2*E + w3*A + w4*N
    dove O,C,E,A,N sono i tratti di personalità (Openness, Conscientiousness,
    Extraversion, Agreeableness, Neuroticism) e w0,...,w4 sono i pesi configurabili.
    """
    
    def __init__(self, weights=None, name="RegressionProcessor"):
        """
        Inizializza il processor di regressione.
        
        Args:
            weights (dict, optional): Dizionario con i pesi {trait: weight}.
                                    Se None, verranno utilizzati pesi dalla configurazione.
            name (str): Nome del processor.
        """
        super().__init__(name=name)
        
        # Usa i pesi dalla configurazione se non specificati
        if weights is None:
            self.weights = {    
                'openness': REGRESSION_WEIGHT_OPENNESS,                    # w0
                'conscientiousness': REGRESSION_WEIGHT_CONSCIENTIOUSNESS, # w1
                'extraversion': REGRESSION_WEIGHT_EXTRAVERSION,           # w2
                'agreeableness': REGRESSION_WEIGHT_AGREEABLENESS,         # w3
                'neuroticism': REGRESSION_WEIGHT_NEUROTICISM              # w4
            }
        else:
            self.weights = weights
            
        logger.info(f"Inizializzato {name} con pesi: {self.weights}")
    
    def calculate_behavior(self, personality_traits):
        """
        Calcola il valore di comportamento dell'utente in base ai tratti di personalità.
        
        Args:
            personality_traits (dict): Dizionario con i tratti di personalità.
            
        Returns:
            float: Valore di comportamento calcolato.
            
        Raises:
            ValueError: Se i tratti di personalità non hanno il formato atteso.
        """
        if not isinstance(personality_traits, dict):
            raise ValueError("I tratti di personalità devono essere forniti come dizionario")
        
        try:
            if isinstance(personality_traits, dict):
                o_value = float(personality_traits.get('openness', 0))
                c_value = float(personality_traits.get('conscientiousness', 0))
                e_value = float(personality_traits.get('extraversion', 0))
                a_value = float(personality_traits.get('agreeableness', 0))
                n_value = float(personality_traits.get('neuroticism', 0))
            else:
                raise ValueError("Formato tratti di personalità non valido")
            
            behavior_value = (
                self.weights['openness'] * o_value +
                self.weights['conscientiousness'] * c_value +
                self.weights['extraversion'] * e_value +
                self.weights['agreeableness'] * a_value +
                self.weights['neuroticism'] * n_value
            )
            
            behavior_value = round(behavior_value, 2)
            
            logger.debug(f"Calcolato valore di comportamento: {behavior_value} per tratti: {personality_traits}")
            return behavior_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Errore nella conversione dei valori: {e}")
            raise ValueError(f"Errore nel calcolo del comportamento: {e}")
    
    def process(self, traits):
        """        
        Calcola direttamente il valore di comportamento usando i tratti di personalità.
        
        Args:
            traits: Tratti come dizionario.
            
        Returns:
            float: Valore di comportamento calcolato.
        """
        if isinstance(traits, dict):
            return self.calculate_behavior(traits)
        else:
            raise ValueError("Formato tratti di personalità non valido")
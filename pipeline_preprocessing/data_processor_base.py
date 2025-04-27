"""
Interfaccia base per le componenti (data_processor) di preprocessing dei dati.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class DataProcessorBase(ABC):
    """
    Interfaccia comune che tutti i componenti della pipeline di preprocessing devono implementare.
    """
    @abstractmethod
    def process_data(self, source_data):
        """
        Elabora i dati di origine e li trasforma nel formato richiesto per il salvataggio.
        
        Args:
            source_data: I dati di origine in qualsiasi formato specificato dall'implementazione
            
        Returns:
            list: Lista di record elaborati pronti per il salvataggio.
        """
        pass
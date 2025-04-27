"""
Interfaccia base per le componenti (processor) di inferenza.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class InferencePipelineBase(ABC):
    """
    Interfaccia comune che tutti i componenti della pipeline di inferenza devono implementare.
    """
    def __init__(self, name=None):
        """
        Inizializzazione il componente della pipeline.
        
        Args:
            name (str, optional): Nome identificativo del componente.
        """
        self.name = name or self.__class__.__name__
        logger.debug(f"Inizializzato componente pipeline: {self.name}")
    
    @abstractmethod
    def process(self, traits):
        """
        Elabora i dati di input e restituisce il risultato/predizione.
        
        Args:
            traits: I tratti in input utilizzati come variabili indipendenti

        Returns:
            Predizione prodotta dal componente di inferenza.
        """
        pass

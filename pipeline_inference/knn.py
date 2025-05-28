"""
Modulo per l'elaborazione dei dati utilizzando l'algoritmo K-Nearest Neighbors.

Questo modulo fornisce funzionalità per trovare i profili più simili
basati su vari tratti utilizzando l'algoritmo KNN.
"""

import logging
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pipeline_inference.pipeline_inference_base import InferencePipelineBase

logger = logging.getLogger(__name__)

class KNNProcessor(InferencePipelineBase):
    """
    Processor per trovare i vicini più prossimi utilizzando l'algoritmo KNN.
    """
    
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean', weights='uniform', auto_load=True):
        """
        Inizializza il processor KNN.
        
        Args:
            n_neighbors (int): Numero di vicini da trovare.
            algorithm (str): Algoritmo da utilizzare ('auto', 'ball_tree', 'kd_tree', 'brute').
            metric (str): Metrica di distanza da utilizzare.
            weights (str): Ponderazione da utilizzare ('uniform', 'distance').
            auto_load (bool): Se True, tenta di caricare automaticamente un modello esistente.
        """
        super().__init__(name="KNNProcessor")
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights
        self.data = None
        self.pipeline = None
        self.records = None
        
        # Tentativo di caricamento automatico del modello
        if auto_load:
            model_path = self._get_default_model_path()
            if os.path.exists(model_path):
                try:
                    self.load_model(model_path)
                    logger.info("Modello KNN caricato automaticamente")
                except Exception as e:
                    logger.warning(f"Impossibile caricare il modello automaticamente: {e}")
                    self._initialize_new_model()
            else:
                logger.info("Nessun modello KNN esistente trovato, inizializzazione nuovo modello")
                self._initialize_new_model()
        else:
            self._initialize_new_model()
    
    def _get_default_model_path(self):
        """
        Ottiene il percorso predefinito del modello KNN.
        
        Returns:
            str: Percorso del file del modello.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "models"))
        filename = f"knn_model_n-{self.n_neighbors}_a-{self.algorithm}_m-{self.metric}_w-{self.weights}.joblib"
        return os.path.join(models_dir, filename)
    
    def _initialize_new_model(self):
        """
        Inizializza un nuovo modello KNN non addestrato.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                metric=self.metric
            ))
        ])
        logger.info(f"Inizializzato nuovo processor KNN: n_neighbors={self.n_neighbors}, algorithm={self.algorithm}, metric={self.metric}")
    
    def is_trained(self):
        """
        Verifica se il modello è stato addestrato.
        
        Returns:
            bool: True se il modello è addestrato, False altrimenti.
        """
        return (self.pipeline is not None and 
                hasattr(self.pipeline.named_steps['knn'], 'n_samples_fit_') and
                self.data is not None and 
                self.records is not None)
    
    def save_model(self, path=None):
        """
        Salva il modello KNN e i dati di addestramento su disco.
        
        Args:
            path (str, optional): Percorso dove salvare il modello. Se None, usa un nome predefinito.
            
        Returns:
            str: Percorso dove è stato salvato il modello.
            
        Raises:
            RuntimeError: Se il modello non è stato addestrato.
        """
        if not self.is_trained():
            raise RuntimeError("Nessun modello addestrato da salvare. Addestrare prima il modello.")
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "models"))
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info(f"Creata directory {models_dir}")
            
            if path is None:
                path = self._get_default_model_path()
            
            # Salvataggio modello e dati necessari per l'inferenza
            model_data = {
                'pipeline': self.pipeline,
                'data': self.data,
                'records': self.records,
                'n_neighbors': self.n_neighbors,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'weights': self.weights
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Modello KNN e dati salvati in: {path}")
            return path
        except Exception as e:
            logger.error(f"Errore durante il salvataggio del modello: {e}")
            raise
    
    def load_model(self, path):
        """
        Carica un modello KNN precedentemente salvato.
        
        Args:
            path (str): Percorso del modello da caricare.
            
        Raises:
            FileNotFoundError: Se il file del modello non esiste.
        """
        try:
            model_data = joblib.load(path)
            
            # Caricamento componenti necessari
            self.pipeline = model_data['pipeline']
            self.data = model_data['data']
            self.records = model_data['records']
            self.n_neighbors = model_data.get('n_neighbors', self.n_neighbors)
            self.algorithm = model_data.get('algorithm', self.algorithm)
            self.metric = model_data.get('metric', self.metric)
            self.weights = model_data.get('weights', self.weights)
            
            logger.info(f"Modello KNN caricato da: {path}")
        except FileNotFoundError:
            logger.error(f"File del modello non trovato: {path}")
            raise
        except Exception as e:
            logger.error(f"Errore durante il caricamento del modello: {e}")
            raise
    
    def fit(self, records):
        """
        Addestra il modello KNN sui dati forniti.
        
        Args:
            records (list): Lista di record con tratti di personalità.
            
        Returns:
            self: Il processor stesso.
            
        Raises:
            ValueError: Se i record non hanno il formato atteso.
        """
        try:
            if not records:
                raise ValueError("La lista dei record non può essere vuota")
            
            self.records = records
            personality_vectors = []
            
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
                personality_vectors.append(vector)
            
            self.data = np.array(personality_vectors)
            
            # Se il modello non è inizializzato, crealo
            if self.pipeline is None:
                self._initialize_new_model()
            
            self.pipeline.fit(self.data)
            self.save_model()
            
            logger.info(f"Modello KNN addestrato su {len(records)} record")
            return self
            
        except Exception as e:
            logger.error(f"Errore durante l'addestramento del modello KNN: {e}")
            raise
    
    def find_neighbors(self, personality_traits):
        """
        Trova i vicini più prossimi in base ai tratti di personalità.
        
        Args:
            personality_traits (dict o array): Tratti di personalità come dizionario o array numpy.
            
        Returns:
            list: Lista dei vicini trovati con le loro distanze.
            
        Raises:
            RuntimeError: Se il modello non è stato addestrato.
        """
        if not self.is_trained():
            raise RuntimeError("Il modello KNN deve essere addestrato prima di cercare i vicini")
        
        try:
            if isinstance(personality_traits, dict):
                query_vector = np.array([
                    personality_traits.get('extraversion', 0),
                    personality_traits.get('agreeableness', 0),
                    personality_traits.get('conscientiousness', 0),
                    personality_traits.get('neuroticism', 0),
                    personality_traits.get('openness', 0)
                ]).reshape(1, -1)
            else:
                raise ValueError("Formato tratti di personalità non valido")
            
            k = min(self.n_neighbors, len(self.records))
            
            distances, indices = self.pipeline.named_steps['knn'].kneighbors(
                self.pipeline.named_steps['scaler'].transform(query_vector), 
                n_neighbors=k
            )
            
            neighbors = []
            for i, idx in enumerate(indices[0]):
                neighbor = self.records[idx].copy()
                neighbor['similarity_distance'] = float(distances[0][i])
                neighbors.append(neighbor)
            logger.info(f"Trovati {len(neighbors)} vicini")

            return neighbors
            
        except Exception as e:
            logger.error(f"Errore durante la ricerca dei vicini: {e}")
            raise
    
    def process(self, traits):
        """
        Trova la criticità media dei vicini più simili in base ai tratti di personalità.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            
        Returns:
            float: Media dell'indice di criticità dei vicini
            
        Raises:
            ValueError: Se i dati di input non hanno il formato corretto
            RuntimeError: Se il modello non è addestrato
        """
        if not isinstance(traits, dict):
            raise ValueError("Formato tratti di personalità non valido")
        
        if not self.is_trained():
            raise RuntimeError("Il modello KNN deve essere addestrato prima di effettuare predizioni")
        
        try:
            neighbors = self.find_neighbors(traits)
            
            for i, record in enumerate(neighbors):
                logger.info(f"Profilo {i+1}: "
                        f"Extraversion: {record.get('personality_traits', {}).get('extraversion', 'N/A')}, "
                        f"Agreeableness: {record.get('personality_traits', {}).get('agreeableness', 'N/A')}, "
                        f"Conscientiousness: {record.get('personality_traits', {}).get('conscientiousness', 'N/A')}, "
                        f"Neuroticism: {record.get('personality_traits', {}).get('neuroticism', 'N/A')}, "
                        f"Openness: {record.get('personality_traits', {}).get('openness', 'N/A')}, "
                        f"Criticality: {record.get('criticality_index', 'N/A')}, "
                        f"Similarity: {record.get('similarity_distance', 'N/A')}")
            
            # Calcolo media dell'indice di criticità dei vicini
            criticality_values = [record.get('criticality_index', 0) for record in neighbors if record.get('criticality_index') is not None]
            avg_criticality = sum(criticality_values) / len(criticality_values) if criticality_values else 0
            
            logger.info(f"Processo KNN completato. Media indice di criticità: {avg_criticality:.2f}")
            
            return round(avg_criticality, 2)
        except Exception as e:
            logger.error(f"Errore durante il processo KNN: {e}")
            raise

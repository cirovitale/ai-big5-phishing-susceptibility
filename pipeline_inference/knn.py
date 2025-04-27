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
    
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean', weights='uniform', model_path=None):
        """
        Inizializza il processor KNN.
        
        Args:
            n_neighbors (int): Numero di vicini da trovare.
            algorithm (str): Algoritmo da utilizzare ('auto', 'ball_tree', 'kd_tree', 'brute').
            metric (str): Metrica di distanza da utilizzare.
            weights (str): Ponderazione da utilizzare ('uniform', 'distance').
            model_path (str, optional): Percorso del modello salvato da caricare.
        """
        super().__init__(name="KNNProcessor")
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights
        self.data = None
        self.pipeline = None
        self.records = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', NearestNeighbors(
                    n_neighbors=self.n_neighbors,
                    algorithm=self.algorithm,
                    metric=self.metric
                ))
            ])
            logger.info(f"Inizializzato processor KNN: n_neighbors={n_neighbors}, algorithm={algorithm}, metric={metric}")
    
    def save_model(self, path=None):
        """
        Salva il modello KNN su disco.
        
        Args:
            path (str, optional): Percorso dove salvare il modello. Se None, usa un nome predefinito.
            
        Returns:
            str: Percorso dove è stato salvato il modello.
            
        Raises:
            RuntimeError: Se il modello non è stato addestrato.
        """
        if self.pipeline is None:
            raise RuntimeError("Nessun modello da salvare. Addestrare prima il modello.")
        
        try:
            # Directory del file corrente
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Salgo di un livello e punto alla cartella 'models'
            models_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "models"))
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                logger.info(f"Creata directory {models_dir}")
            
            if path is None:
                filename = f"knn_model_n-{self.n_neighbors}_a-{self.algorithm}_m-{self.metric}_w-{self.weights}.joblib"
                path = os.path.join(models_dir, filename)
            
            joblib.dump(self.pipeline, path)
            logger.info(f"Modello KNN salvato in: {path}")
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
            self.pipeline = joblib.load(path)
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
            
            # Addestra la pipeline
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
            tuple: (distances, indices) delle distanze e indici dei vicini trovati.
            
        Raises:
            RuntimeError: Se il modello non è stato addestrato.
        """
        if self.pipeline is None or self.data is None or self.records is None:
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
            dict: Media dell'indice di criticità dei vicini
            
        Raises:
            ValueError: Se i dati di input non hanno il formato corretto
        """
        if isinstance(traits, dict):
            neighbors = self.find_neighbors(traits)
        else:
            raise ValueError("Formato tratti di personalità non valido")
        try:
            for i, record in enumerate(neighbors):
                logger.info(f"Profilo {i+1}: "
                        # f"Country: {record.get('demographic_traits', {}).get('Country', 'N/A')}, "
                        # f"Gender: {record.get('demographic_traits', {}).get('Gender', 'N/A')}, "
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

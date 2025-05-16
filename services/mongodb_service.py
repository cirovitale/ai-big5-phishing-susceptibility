"""
Modulo per l'interazione con MongoDB.

Questo modulo fornisce funzionalità per la connessione a MongoDB
e l'inserimento o aggiornamento dei dati elaborati.
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION_DATASET, MONGODB_COLLECTION_INFERENCE

logger = logging.getLogger(__name__)

class MongoDBService:
    """
    Classe per gestire le interazioni con MongoDB.
    """
    
    def __init__(self):
        """
        Inizializza la classe MongoDBService.
        """
        self.client = None
        self.db = None
        self.dataset_collection = None
        self.inference_collection = None
        self.connect()
        self.select_database_and_collections()
    
    def connect(self):
        """
        Stabilisce una connessione con MongoDB.
        
        Returns:
            MongoClient: Client MongoDB connesso.
        
        Raises:
            ConnectionFailure: Se non è possibile connettersi al server MongoDB.
        """
        try:
            self.client = MongoClient(MONGODB_URI)
            # self.client.admin.command('ping')
            
            logger.info("Connessione a MongoDB stabilita con successo")
            return self.client
        
        except ConnectionFailure as e:
            logger.error(f"Impossibile connettersi a MongoDB: {e}")
            raise
    
    def select_database_and_collections(self):
        """
        Seleziona il database e le collezioni specificate.
        
        Deve essere chiamato dopo connect().
        
        Returns:
            tuple: (dataset_collection, inference_collection) Le collezioni MongoDB selezionate.
        
        Raises:
            RuntimeError: Se chiamato prima della connessione.
            OperationFailure: Se si verificano errori nell'accesso al database/collezioni.
        """
        if self.client is None:
            logger.error("Tentativo di selezionare database senza connessione")
            raise RuntimeError("È necessario chiamare connect() prima di select_database_and_collections()")
        
        try:
            self.db = self.client[MONGODB_DB]
            self.dataset_collection = self.db[MONGODB_COLLECTION_DATASET]
            self.inference_collection = self.db[MONGODB_COLLECTION_INFERENCE]
            
            logger.info(f"Database '{MONGODB_DB}' e collezioni '{MONGODB_COLLECTION_DATASET}', '{MONGODB_COLLECTION_INFERENCE}' selezionati")
            return self.dataset_collection, self.inference_collection
        
        except OperationFailure as e:
            logger.error(f"Errore nell'accesso a database o collezioni: {e}")
            raise
    
    def save_data(self, processed_data):
        """
        Salva i dati elaborati in MongoDB nella collezione dataset.
        
        Args:
            processed_data (list): Lista di dizionari contenenti i dati da salvare.
        
        Returns:
            tuple: (num_inserted, num_updated) conteggio dei record inseriti e aggiornati.
        
        Raises:
            RuntimeError: Se la collezione non è stata selezionata.
        """
        if self.dataset_collection is None:
            logger.error("Collezione dataset non selezionata")
            raise RuntimeError("Collezione dataset non selezionata. Chiamare select_database_and_collections() prima.")
    
        num_inserted = 0
        num_updated = 0
        
        for record in processed_data:
            result = self.dataset_collection.insert_one(record)
            
            if result.inserted_id:
                logger.debug(f"Inserito nuovo record con ID: {result.inserted_id}")
                num_inserted += 1
            else:
                logger.warning(f"Impossibile inserire il record")
        
        logger.info(f"Operazione completata: {num_inserted} record inseriti, {num_updated} record aggiornati")
        return num_inserted, num_updated
    
    def get_all_records(self):
        """
        Recupera tutti i record dalla collezione dataset di MongoDB.
        
        Returns:
            list: Lista di tutti i record nella collezione.
        """
        try:
            if self.dataset_collection is None:
                logger.error("Collezione dataset non selezionata")
                raise RuntimeError("Collezione dataset non selezionata. Chiamare select_database_and_collections() prima.")
            
            # Recupera tutti i record
            records = list(self.dataset_collection.find({}))

            for record in records:
                record.pop('raw_data', None)

            logger.info(f"Recuperati {len(records)} record da MongoDB")
            return records
        
        except Exception as e:
            logger.error(f"Errore durante il recupero dei record da MongoDB: {e}")
            raise
    
    def close_connection(self):
        """
        Chiude la connessione a MongoDB.
        """
        if self.client is not None:
            self.client.close()
            logger.info("Connessione a MongoDB chiusa")

    def save_prediction(self, prediction_data):
        """
        Salva una predizione in MongoDB nella collezione inference.
        
        Args:
            prediction_data (dict): Dati della predizione
        
        Returns:
            str: ID del documento inserito
        
        Raises:
            RuntimeError: Se la collezione non è stata selezionata
        """
        if self.inference_collection is None:
            logger.error("Collezione inference non selezionata")
            raise RuntimeError("Collezione inference non selezionata. Chiamare select_database_and_collections() prima.")
        
        try:
            # Salva la predizione
            result = self.inference_collection.insert_one(prediction_data)
            
            if result.inserted_id:
                logger.debug(f"Predizione salvata con ID: {result.inserted_id}")
                return str(result.inserted_id)
            else:
                logger.warning(f"Impossibile salvare la predizione")
                return None
        
        except Exception as e:
            logger.error(f"Errore durante il salvataggio della predizione: {e}")
            raise

    def get_record_count(self):
        """
        Restituisce il numero di record nella collezione dataset.
        
        Returns:
            int: Numero di record nel dataset
        """
        try:
            return self.db[self.dataset_collection].count_documents({})
        except Exception as e:
            logger.error(f"Errore durante il conteggio dei record: {e}")
            return 0
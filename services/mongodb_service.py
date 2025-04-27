"""
Modulo per l'interazione con MongoDB.

Questo modulo fornisce funzionalità per la connessione a MongoDB
e l'inserimento o aggiornamento dei dati elaborati.
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

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
        self.collection = None
        self.connect()
        self.select_database_and_collection()
    
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
    
    def select_database_and_collection(self):
        """
        Seleziona il database e la collezione specificati.
        
        Deve essere chiamato dopo connect().
        
        Returns:
            Collection: La collezione MongoDB selezionata.
        
        Raises:
            RuntimeError: Se chiamato prima della connessione.
            OperationFailure: Se si verificano errori nell'accesso al database/collezione.
        """
        if self.client is None:
            logger.error("Tentativo di selezionare database senza connessione")
            raise RuntimeError("È necessario chiamare connect() prima di select_database_and_collection()")
        
        try:
            self.db = self.client[MONGODB_DB]
            self.collection = self.db[MONGODB_COLLECTION]
            
            logger.info(f"Database '{MONGODB_DB}' e collezione '{MONGODB_COLLECTION}' selezionati")
            return self.collection
        
        except OperationFailure as e:
            logger.error(f"Errore nell'accesso a database o collezione: {e}")
            raise
    
    def save_data(self, processed_data):
        """
        Salva i dati elaborati in MongoDB.
        
        Args:
            processed_data (list): Lista di dizionari contenenti i dati da salvare.
        
        Returns:
            tuple: (num_inserted, num_updated) conteggio dei record inseriti e aggiornati.
        
        Raises:
            RuntimeError: Se la collezione non è stata selezionata.
        """
        if self.collection is None:
            logger.error("Collezione non selezionata")
            raise RuntimeError("Collezione non selezionata. Chiamare select_database_and_collection() prima.")
    
        num_inserted = 0
        num_updated = 0
        
        for record in processed_data:
            result = self.collection.insert_one(record)
            
            if result.inserted_id:
                logger.debug(f"Inserito nuovo record con ID: {result.inserted_id}")
                num_inserted += 1
            else:
                logger.warning(f"Impossibile inserire il record")
        
        logger.info(f"Operazione completata: {num_inserted} record inseriti, {num_updated} record aggiornati")
        return num_inserted, num_updated
    
    def get_all_records(self):
        """
        Recupera tutti i record dalla collezione MongoDB.
        
        Returns:
            list: Lista di tutti i record nella collezione.
        """
        try:
            if self.collection is None:
                logger.error("Collezione non selezionata")
                raise RuntimeError("Collezione non selezionata. Chiamare select_database_and_collection() prima.")
            
            # Recupera tutti i record
            records = list(self.collection.find({}))

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


    
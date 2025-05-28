"""
Modulo per l'interazione con MongoDB.

Questo modulo fornisce funzionalità per la connessione a MongoDB
e l'inserimento o aggiornamento dei dati elaborati.
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION_DATASET, MONGODB_COLLECTION_INFERENCE, MONGODB_COLLECTION_DT
import datetime

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
        self.digital_twin_collection = None
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
            tuple: (dataset_collection, inference_collection, digital_twin_collection) Le collezioni MongoDB selezionate.
        
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
            self.digital_twin_collection = self.db[MONGODB_COLLECTION_DT]
            
            logger.info(f"Database '{MONGODB_DB}' e collezioni '{MONGODB_COLLECTION_DATASET}', '{MONGODB_COLLECTION_INFERENCE}', '{MONGODB_COLLECTION_DT}' selezionati")
            return self.dataset_collection, self.inference_collection, self.digital_twin_collection
        
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
    
    def get_all_dataset_records(self):
        """
        Recupera tutti i record dal dataset di personalità dalla collezione dataset di MongoDB.
        
        Returns:
            list: Lista di tutti i record del dataset nella collezione, con i campi 'raw_data' rimossi per efficienza.
            
        Raises:
            RuntimeError: Se la collezione dataset non è stata selezionata.
            Exception: Se si verificano errori durante il recupero dei dati da MongoDB.
        """
        try:
            if self.dataset_collection is None:
                logger.error("Collezione dataset non selezionata")
                raise RuntimeError("Collezione dataset non selezionata. Chiamare select_database_and_collections() prima.")
            
            # Recupera tutti i record
            records = list(self.dataset_collection.find({}))

            for record in records:
                record.pop('raw_data', None)

            logger.info(f"Recuperati {len(records)} record del dataset da MongoDB")
            return records
        
        except Exception as e:
            logger.error(f"Errore durante il recupero dei record del dataset da MongoDB: {e}")
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
        Restituisce il numero di record del dataset nella collezione dataset.
        
        Returns:
            int: Numero di record del dataset nella collezione
            
        Raises:
            RuntimeError: Se la collezione dataset non è stata selezionata
        """
        try:
            if self.dataset_collection is None:
                logger.error("Collezione dataset non selezionata")
                raise RuntimeError("Collezione dataset non selezionata. Chiamare select_database_and_collections() prima.")
            
            count = self.dataset_collection.count_documents({})
            logger.debug(f"Conteggio record del dataset: {count}")
            return count
        except Exception as e:
            logger.error(f"Errore durante il conteggio dei record del dataset: {e}")
            return 0

    def register_digital_twin(self, digital_twin_data):
        """
        Registra un nuovo digital twin in MongoDB.
        
        Args:
            digital_twin_data (dict): Dati del digital twin contenenti:
                - cf (str): Codice fiscale
                - first_name (str): Nome
                - last_name (str): Cognome  
                - traits (dict): Tratti di personalità Big5
                - creation_datetime (datetime, opzionale): Data/ora di creazione
                - last_update_datetime (datetime, opzionale): Data/ora ultimo aggiornamento
                - last_training_datetime (datetime, opzionale): Data/ora ultimo training
        
        Returns:
            str: ID del documento inserito
        
        Raises:
            RuntimeError: Se la collezione digital twin non è stata selezionata
            ValueError: Se i dati obbligatori sono mancanti
        """
        if self.digital_twin_collection is None:
            logger.error("Collezione digital twin non selezionata")
            raise RuntimeError("Collezione digital twin non selezionata. Chiamare select_database_and_collections() prima.")
        
        required_fields = ['cf', 'first_name', 'last_name', 'traits']
        for field in required_fields:
            if field not in digital_twin_data or not digital_twin_data[field]:
                raise ValueError(f"Campo obbligatorio mancante: {field}")
        
        required_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        traits = digital_twin_data['traits']
        for trait in required_traits:
            if trait not in traits:
                raise ValueError(f"Tratto obbligatorio mancante: {trait}")
        
        try:
            # Verifica se esiste già un digital twin con lo stesso CF
            existing_dt = self.digital_twin_collection.find_one({"cf": digital_twin_data['cf']})
            if existing_dt:
                raise ValueError(f"Digital twin con CF {digital_twin_data['cf']} già esistente")
            
            # Prepara i dati con timestamps
            current_time = datetime.datetime.now()
            dt_record = {
                "cf": digital_twin_data['cf'],
                "first_name": digital_twin_data['first_name'],
                "last_name": digital_twin_data['last_name'],
                "traits": digital_twin_data['traits'],
                "creation_datetime": digital_twin_data.get('creation_datetime', current_time),
                "last_update_datetime": digital_twin_data.get('last_update_datetime', current_time),
                "last_training_datetime": digital_twin_data.get('last_training_datetime', None)
            }
            
            result = self.digital_twin_collection.insert_one(dt_record)
            
            if result.inserted_id:
                logger.info(f"Digital twin registrato con successo per CF: {digital_twin_data['cf']}, ID: {result.inserted_id}")
                return str(result.inserted_id)
            else:
                logger.warning(f"Impossibile registrare il digital twin per CF: {digital_twin_data['cf']}")
                return None
        
        except Exception as e:
            logger.error(f"Errore durante la registrazione del digital twin: {e}")
            raise

    def get_digital_twin_by_cf(self, cf):
        """
        Recupera un digital twin tramite codice fiscale.
        
        Args:
            cf (str): Codice fiscale
        
        Returns:
            dict: Dati del digital twin trovato, None se non trovato
        
        Raises:
            RuntimeError: Se la collezione digital twin non è stata selezionata
        """
        if self.digital_twin_collection is None:
            logger.error("Collezione digital twin non selezionata")
            raise RuntimeError("Collezione digital twin non selezionata. Chiamare select_database_and_collections() prima.")
        
        try:
            digital_twin = self.digital_twin_collection.find_one({"cf": cf})
            
            if digital_twin:
                logger.debug(f"Digital twin trovato per CF: {cf}")
                return digital_twin
            else:
                logger.warning(f"Nessun digital twin trovato per CF: {cf}")
                return None
        
        except Exception as e:
            logger.error(f"Errore durante il recupero del digital twin per CF {cf}: {e}")
            raise

    def update_digital_twin_last_training(self, cf):
        """
        Aggiorna la data dell'ultimo training per un digital twin.
        
        Args:
            cf (str): Codice fiscale del digital twin
        
        Returns:
            bool: True se l'aggiornamento è avvenuto con successo, False altrimenti
        
        Raises:
            RuntimeError: Se la collezione digital twin non è stata selezionata
        """
        if self.digital_twin_collection is None:
            logger.error("Collezione digital twin non selezionata")
            raise RuntimeError("Collezione digital twin non selezionata. Chiamare select_database_and_collections() prima.")
        
        try:
            current_time = datetime.datetime.now()
            result = self.digital_twin_collection.update_one(
                {"cf": cf},
                {
                    "$set": {
                        "last_training_datetime": current_time,
                        "last_update_datetime": current_time
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Data ultimo training aggiornata per CF: {cf}")
                return True
            else:
                logger.warning(f"Nessun digital twin aggiornato per CF: {cf}")
                return False
        
        except Exception as e:
            logger.error(f"Errore durante l'aggiornamento dell'ultimo training per CF {cf}: {e}")
            raise
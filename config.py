"""
Modulo di configurazione dell'applicazione.

Questo modulo si occupa di caricare le variabili d'ambiente e fornire le configurazioni necessarie per l'applicazione.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Dataset Preprocessing
DATASET_PATH = os.getenv('DATASET_PATH')
# DATASET_PATH = os.path.join(dataset_dir, 'Dataset.xlsx') if dataset_dir else None

# MongoDB
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION_DATASET = os.getenv('MONGODB_COLLECTION_DATASET')
MONGODB_COLLECTION_INFERENCE = os.getenv('MONGODB_COLLECTION_INFERENCE')
SCALING_TYPE = os.getenv('SCALING_TYPE')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL')

def validate_configuration():
    """
    Verifica che tutte le variabili di configurazione necessarie siano definite.
    
    Returns:
        bool: True se la configurazione è valida, False altrimenti.
    """
    required_config = {
        'DATASET_PATH': DATASET_PATH,
        'MONGODB_URI': MONGODB_URI,
        'MONGODB_DB': MONGODB_DB,
        'MONGODB_COLLECTION_DATASET': MONGODB_COLLECTION_DATASET,
        'MONGODB_COLLECTION_INFERENCE': MONGODB_COLLECTION_INFERENCE,
        'SCALING_TYPE': SCALING_TYPE
    }
    
    for key, value in required_config.items():
        if not value:
            print(f"Errore: La variabile di configurazione '{key}' non è definita.")
            return False
    
    return True

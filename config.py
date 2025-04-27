"""
Modulo di configurazione dell'applicazione.

Questo modulo si occupa di caricare le variabili d'ambiente e fornire le configurazioni necessarie per l'applicazione.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Google API
SCOPES = ['https://www.googleapis.com/auth/forms.readonly']
GOOGLE_FORM_ID = os.getenv('GOOGLE_FORM_ID')
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')

# Dataset Preprocessing
DATASET_PATH = os.getenv('DATASET_PATH')

# MongoDB
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL')

def validate_configuration():
    """
    Verifica che tutte le variabili di configurazione necessarie siano definite.
    
    Returns:
        bool: True se la configurazione è valida, False altrimenti.
    """
    required_config = {
        'GOOGLE_FORM_ID': GOOGLE_FORM_ID,
        'DATASET_PATH': DATASET_PATH,
        'GOOGLE_SERVICE_ACCOUNT_FILE': GOOGLE_SERVICE_ACCOUNT_FILE,
        'MONGODB_URI': MONGODB_URI,
        'MONGODB_DB': MONGODB_DB,
        'MONGODB_COLLECTION': MONGODB_COLLECTION
    }
    
    for key, value in required_config.items():
        if not value:
            print(f"Errore: La variabile di configurazione '{key}' non è definita.")
            return False
    
    return True

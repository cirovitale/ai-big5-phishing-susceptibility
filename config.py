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
MONGODB_COLLECTION_DT = os.getenv('MONGODB_COLLECTION_DT', 'digital-twin')
SCALING_TYPE = os.getenv('SCALING_TYPE')

#OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensemble Weights
ENSEMBLE_WEIGHT_KNN = float(os.getenv('ENSEMBLE_WEIGHT_KNN', '1'))
ENSEMBLE_WEIGHT_REGRESSION = float(os.getenv('ENSEMBLE_WEIGHT_REGRESSION', '1'))
ENSEMBLE_WEIGHT_LLM = float(os.getenv('ENSEMBLE_WEIGHT_LLM', '1'))
ENSEMBLE_WEIGHT_DL = float(os.getenv('ENSEMBLE_WEIGHT_DL', '1'))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL')

def validate_configuration():
    """
    Verifica che tutte le variabili di configurazione necessarie siano definite.
    
    Returns:
        bool: True se la configurazione è valida, False altrimenti.
    """
    # Variabili che non possono essere None o stringhe vuote
    required_string_config = {
        'DATASET_PATH': DATASET_PATH,
        'MONGODB_URI': MONGODB_URI,
        'MONGODB_DB': MONGODB_DB,
        'MONGODB_COLLECTION_DATASET': MONGODB_COLLECTION_DATASET,
        'MONGODB_COLLECTION_INFERENCE': MONGODB_COLLECTION_INFERENCE,
        'MONGODB_COLLECTION_DT': MONGODB_COLLECTION_DT,
        'SCALING_TYPE': SCALING_TYPE,
        'OPENAI_API_KEY': OPENAI_API_KEY
    }
    
    # Variabili numeriche che possono essere 0 ma non None
    required_numeric_config = {
        'ENSEMBLE_WEIGHT_KNN': ENSEMBLE_WEIGHT_KNN,
        'ENSEMBLE_WEIGHT_REGRESSION': ENSEMBLE_WEIGHT_REGRESSION,
        'ENSEMBLE_WEIGHT_LLM': ENSEMBLE_WEIGHT_LLM,
        'ENSEMBLE_WEIGHT_DL': ENSEMBLE_WEIGHT_DL
    }
    
    # Controlla le variabili stringa
    for key, value in required_string_config.items():
        if not value:
            print(f"Errore: La variabile di configurazione '{key}' non è definita.")
            return False
    
    # Controlla le variabili numeriche (possono essere 0)
    for key, value in required_numeric_config.items():
        if value is None:
            print(f"Errore: La variabile di configurazione '{key}' non è definita.")
            return False
    
    return True

"""
Server Flask per il sistema di predizione di suscettibilità al phishing
"""
import logging
import sys
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import validate_configuration, LOG_LEVEL, MONGODB_COLLECTION_DT, ENSEMBLE_WEIGHT_KNN, ENSEMBLE_WEIGHT_REGRESSION, ENSEMBLE_WEIGHT_LLM, ENSEMBLE_WEIGHT_DL
from pipeline_inference.dl import DLProcessor
from pipeline_inference.llm import LLMProcessor
from services.mongodb_service import MongoDBService
from pipeline_preprocessing.dataset_excel import ExcelDataProcessor
from pipeline_inference.knn import KNNProcessor
from pipeline_inference.regression import RegressionProcessor
from pipeline_inference.ensemble import EnsembleProcessor
from pipeline_testing.tester import Tester
import datetime

# Configurazione logging
LOG_FILE = os.getenv('LOG_FILE', 'app.log')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)]
)

logger = logging.getLogger(__name__)

# Inizializzazione servizi
mongodb_service = MongoDBService()

CRITICALITY_THRESHOLD = float(os.getenv("CRITICALITY_THRESHOLD", "0.5"))

app = Flask(__name__)
CORS(app) 

def execute_extraction_preprocessing_pipeline(process_form=False, process_excel=True):
    """
    Esegue l'estrazione e preprocessing dei dati dal form Google e/o dataset Excel.
    
    Args:
        process_form (bool): Se True, elabora i dati dal form Google
        process_excel (bool): Se True, elabora i dati dal file Excel
        
    Returns:
        list: Record elaborati pronti per l'inferenza
    """
    logger.info("Avvio estrazione e preprocessamento dati")
    all_processed_data = []
    
    try:
        # Elaborazione dati dal form Google
        if process_form:
            # TODO
            pass
        
        # Elaborazione dati dal dataset Excel
        if process_excel:
            logger.info("Elaborazione dati dal file Excel")
            excel_processor = ExcelDataProcessor()
            
            # Elaborazione dati Excel
            excel_processed_data = excel_processor.process_data()
            all_processed_data.extend(excel_processed_data)
            logger.info(f"Aggiunti {len(excel_processed_data)} record dal file Excel")
        
        if all_processed_data:
            logger.info("Salvataggio dei dati elaborati in MongoDB")
            mongodb_service.save_data(all_processed_data)
        
        logger.info(f"Estrazione completata: {len(all_processed_data)} record totali elaborati")
        return all_processed_data
        
    except Exception as e:
        logger.error(f"Errore durante l'estrazione dei dati: {e}", exc_info=True)
        return []

def execute_training_pipeline():
    """
    Esegue la pipeline di addestramento per i modelli KNN e DL utilizzando solo il training set.
    
    Questa funzione addestra i modelli utilizzando solo i dati del training set (validation=False)
    disponibili in MongoDB e salva i modelli addestrati nella directory /models per uso futuro.
    
    Returns:
        dict: Risultati dell'addestramento con informazioni sui modelli addestrati
        
    Raises:
        Exception: Se si verificano errori durante l'addestramento
    """
    logger.info("Avvio pipeline di addestramento")
    
    training_results = {
        'status': 'success',
        'models_trained': [],
        'errors': [],
        'dataset_info': {}
    }
    
    try:
        # Recupera i dati del training set da MongoDB
        data = mongodb_service.get_training_dataset_records()
        if not data:
            error_msg = "Nessun dato del training set disponibile in MongoDB per l'addestramento"
            logger.error(error_msg)
            training_results['status'] = 'error'
            training_results['errors'].append(error_msg)
            return training_results
        
        # Ottieni informazioni sulla divisione train/test
        split_info = mongodb_service.get_train_test_split_info()
        training_results['dataset_info'] = split_info
        
        logger.info(f"Recuperati {len(data)} record del training set da MongoDB per l'addestramento")
        logger.info(f"Training set: {split_info.get('training_records', 0)} record ({split_info.get('training_percentage', 0):.1f}%)")
        
        # Addestramento KNN
        if ENSEMBLE_WEIGHT_KNN > 0:
            try:
                logger.info("Avvio addestramento modello KNN")
                knn_processor = KNNProcessor(auto_load=False)  # Non caricare modello esistente
                knn_processor.fit(data)
                training_results['models_trained'].append('KNN')
                logger.info("Addestramento KNN completato con successo")
            except Exception as e:
                error_msg = f"Errore durante l'addestramento KNN: {e}"
                logger.error(error_msg, exc_info=True)
                training_results['errors'].append(error_msg)
        
        # Addestramento DL
        if ENSEMBLE_WEIGHT_DL > 0:
            try:
                logger.info("Avvio addestramento modello DL")
                dl_processor = DLProcessor(auto_load=False)  # Non caricare modello esistente
                dl_processor.fit(data)
                training_results['models_trained'].append('DL')
                logger.info("Addestramento DL completato con successo")
            except Exception as e:
                error_msg = f"Errore durante l'addestramento DL: {e}"
                logger.error(error_msg, exc_info=True)
                training_results['errors'].append(error_msg)
        
        # Status Finale
        if training_results['errors']:
            if training_results['models_trained']:
                training_results['status'] = 'partial_success'
                logger.warning("Addestramento completato parzialmente con alcuni errori")
            else:
                training_results['status'] = 'error'
                logger.error("Addestramento fallito per tutti i modelli")
        else:
            logger.info("Addestramento completato con successo per tutti i modelli")
        
        return training_results
        
    except Exception as e:
        error_msg = f"Errore critico durante la pipeline di addestramento: {e}"
        logger.error(error_msg, exc_info=True)
        training_results['status'] = 'error'
        training_results['errors'].append(error_msg)
        return training_results

def execute_inference_pipeline(query_traits, sel_question):
    """
    Esegue la pipeline di inferenza sui tratti del soggetto in esame.
    
    Carica i modelli precedentemente addestrati e li utilizza per fare predizioni.
    Se i modelli non esistono, avvia automaticamente l'addestramento utilizzando
    i dati del dataset di personalità disponibili in MongoDB.
    
    Args:
        query_traits (dict): Tratti di personalità su cui inferire
        sel_question: Domanda da utilizzare per il LLM 
        
    Returns:
        tuple: (float, dict) - Valore finale calcolato dall'ensemble e predizioni dei singoli modelli
        
    Raises:
        Exception: Se si verificano errori durante l'inferenza
    """
    logger.info("Avvio pipeline di inference")

    predictions = {}
    question = sel_question
    predicted_behaviour = None
    
    # KNN
    if ENSEMBLE_WEIGHT_KNN > 0:
        try:
            knn_processor = KNNProcessor(auto_load=True)
            
            # Verifica se il modello è addestrato
            if not knn_processor.is_trained():
                logger.warning("Modello KNN non addestrato, avvio addestramento automatico")
                data = mongodb_service.get_training_dataset_records()
                if not data:
                    raise RuntimeError("Nessun dato del dataset disponibile per l'addestramento automatico del modello KNN")
                knn_processor.fit(data)
            
            knn_result = knn_processor.process(query_traits)
            logger.info(f"KNN result: {knn_result}")
            predictions['KNN'] = knn_result
        except Exception as e:
            logger.error(f"Errore durante l'inferenza KNN: {e}", exc_info=True)
            predictions['KNN'] = None
    
    # Regressione
    if ENSEMBLE_WEIGHT_REGRESSION > 0:
        try:
            regression_processor = RegressionProcessor()
            regression_result = regression_processor.process(query_traits)
            
            predictions['Regression'] = regression_result
            logger.info(f"REGRESSION RESULT: {regression_result}")
        except Exception as e:
            logger.error(f"Errore durante il calcolo della regressione: {e}", exc_info=True)
            predictions['Regression'] = None
    
    # LLM
    if ENSEMBLE_WEIGHT_LLM > 0:
        try:
            llm_processor = LLMProcessor(question=question)
            llm_processor_result, predicted_behaviour = llm_processor.process(query_traits)
            
            predictions['LLM'] = llm_processor_result
            logger.info(f"LLM RESULT: {llm_processor_result}")
        except Exception as e:
            logger.error(f"Errore durante l'inferenza LLM: {e}", exc_info=True)
            predictions['LLM'] = None
    
    # DL
    if ENSEMBLE_WEIGHT_DL > 0:
        try:
            dl_processor = DLProcessor(auto_load=True)
            
            # Verifica se il modello è addestrato
            if not dl_processor.is_trained():
                logger.warning("Modello DL non addestrato, avvio addestramento automatico")
                data = mongodb_service.get_training_dataset_records()
                if not data:
                    raise RuntimeError("Nessun dato del dataset disponibile per l'addestramento automatico del modello DL")
                dl_processor.fit(data)
            
            dl_result = dl_processor.process(query_traits)
            logger.info(f"DL result: {dl_result}")
            predictions['DL'] = dl_result
        except Exception as e:
            logger.error(f"Errore durante l'inferenza DL: {e}", exc_info=True)
            predictions['DL'] = None
    
    # ENSEMBLE
    try:
        ensemble_processor = EnsembleProcessor()
        final_result = ensemble_processor.process(predictions)
        
        logger.info(f"ENSEMBLE RESULT: {final_result}")
        return final_result, predictions, predicted_behaviour
    except Exception as e:
        logger.error(f"Errore durante il calcolo dell'ensemble: {e}", exc_info=True)
        return 0, predictions

def execute_testing_pipeline(sel_question):
    """
    Esegue la pipeline di testing utilizzando solo i record del test set in MongoDB.
    
    Args:
        sel_question: Domanda da utilizzare per il LLM 

    Returns:
        dict: Metriche di valutazione del modello (classificazione e regressione)
        
    Raises:
        Exception: Se si verificano errori durante il testing
    """
    logger.info("Avvio pipeline di testing")
    
    try:
        # Recupera solo i record del test set
        records = mongodb_service.get_testing_dataset_records()
        if not records:
            logger.error("Nessun record del test set trovato in MongoDB per il testing")
            return None
        
        # Ottieni informazioni sulla divisione train/test
        split_info = mongodb_service.get_train_test_split_info()
        
        logger.info(f"Recuperati {len(records)} record del test set da MongoDB per il testing")
        logger.info(f"Test set: {split_info.get('testing_records', 0)} record ({split_info.get('testing_percentage', 0):.1f}%)")
        
        question = sel_question
        logger.info(f"Query utilizzata per il testing: {question}")
        
        true_values = []
        predictions = []
        
        for i, record in enumerate(records):
            logger.info(f"Elaborazione record test {i+1}/{len(records)}")
            print(f"Elaborazione record test {i+1}/{len(records)}")
            if 'personality_traits' in record:
                traits = record['personality_traits']
                criticality = record.get('criticality_index', 0)
                
                # Esegui l'inferenza per questo record (conserva solo criticità predetta)
                predicted_criticality, t1, t2  = execute_inference_pipeline(traits, question)
                
                predictions.append(predicted_criticality)
                true_values.append(criticality)
        
        tester = Tester(name="ModelTester")
        test_data = {
            'predictions': predictions,
            'true_values': true_values
        }
        
        metrics = tester.process(test_data)
        
        # Aggiungi informazioni sul dataset alle metriche
        metrics['dataset_info'] = split_info
        metrics['test_records_used'] = len(records)
        
        logger.info(f"Testing completato su {len(records)} record del test set.")
        
        # Accesso corretto alle metriche con la nuova struttura
        classification_metrics = metrics['classification']
        regression_metrics = metrics['regression']
        
        logger.info(f"Metriche di Classificazione - Accuracy: {classification_metrics['accuracy']:.4f}, "
                   f"Precision: {classification_metrics['precision']:.4f}, "
                   f"Recall: {classification_metrics['recall']:.4f}, "
                   f"F1-Score: {classification_metrics['f1_score']:.4f}")
        logger.info(f"Metriche di Regressione - MAE: {regression_metrics['mae']:.4f}, "
                   f"MSE: {regression_metrics['mse']:.4f}, "
                   f"RMSE: {regression_metrics['rmse']:.4f}, "
                   f"R²: {regression_metrics['r2_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Errore durante il testing: {e}", exc_info=True)
        return None

def execute_inference_pipeline_by_cf(cf, question):
    """
    Esegue la pipeline di inferenza per un digital twin identificato dal codice fiscale.
    
    Args:
        cf (str): Codice fiscale del digital twin
        question (str): Domanda effettuata al digital twin
        
    Returns:
        tuple: (float, dict, dict) - Valore finale calcolato dall'ensemble, predizioni dei singoli modelli, e dati del digital twin
        
    Raises:
        ValueError: Se il digital twin non viene trovato
        Exception: Se si verificano errori durante l'inferenza
    """
    logger.info(f"Avvio pipeline di inference per digital twin con CF: {cf}. Domanda effettuata: {question}")
    
    try:
        # Recupera digital twin dal database
        digital_twin = mongodb_service.get_digital_twin_by_cf(cf)
        if not digital_twin:
            raise ValueError(f"Digital twin non trovato per CF: {cf}")
        
        # Estrae i tratti di personalità
        traits = digital_twin.get('traits')
        if not traits:
            raise ValueError(f"Tratti di personalità non trovati per il digital twin con CF: {cf}")
        
        logger.info(f"Tratti recuperati per CF {cf}: {traits}")
        
        # Esegue l'inferenza utilizzando i tratti del digital twin
        result, model_predictions, predicted_behaviour = execute_inference_pipeline(traits, question)
        
        logger.info(f"Inferenza completata per digital twin CF {cf}: {result}")
        return result, model_predictions, predicted_behaviour, digital_twin
        
    except Exception as e:
        logger.error(f"Errore durante l'inferenza per digital twin CF {cf}: {e}", exc_info=True)
        raise

@app.route('/', methods=['GET'])
def index():
    """
    Endpoint della home page.
    
    Returns:
        JSON: Informazioni sullo stato del server e gli endpoint disponibili
    """
    logger.info("Richiesta ricevuta per l'endpoint home (/)")
    return jsonify({
        "status": "online",
        "info": "API per la predizione della suscettibilità al phishing basata su tratti di personalità Big5",
        "endpoints": {
            "/extract": "POST - Esegue estrazione e preprocessing dei dati",
            "/training": "POST - Esegue addestramento dei modelli",
            "/predict": "POST - Predice la suscettibilità al phishing a partire da tratti di personalità",
            "/testing": "GET - Esegue test sul modello in base al dataset",
            "/digital-twin/register": "POST - Registra un nuovo digital twin",
            "/digital-twin/predict": "POST - Predice la suscettibilità a partire dal digital twin (CF)", 
            "/digital-twin/<cf>": "GET - Recupera i dati di un digital twin (CF)",
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint per la predizione basata sui tratti di personalità.
    
    Input JSON:
    {
        "traits": {
            "extraversion": float,
            "agreeableness": float,
            "conscientiousness": float,
            "neuroticism": float,
            "openness": float
        },
        "cf": str (opzionale)
        "question_for_prompt": str
    }
    
    Returns:
        JSON: Predizione della suscettibilità al phishing e predizioni dei singoli modelli; comportamento dell'utente identificato dal Digital Twin
    """
    logger.info("Richiesta ricevuta per l'endpoint /predict")
    try:
        data = request.get_json()
        logger.info(f"Dati ricevuti: {json.dumps(data, indent=2)}")
        
        if not data or 'traits' not in data:
            logger.error("Dati mancanti nella richiesta di predizione")
            return jsonify({"error": "Dati mancanti. È richiesto un JSON con il campo 'traits'."}), 400
        
        traits = data['traits']
        required_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        question = data['question_for_prompt']

        for trait in required_traits:
            if trait not in traits:
                logger.error(f"Tratto mancante nella richiesta: {trait}")
                return jsonify({"error": f"Tratto mancante: {trait}"}), 400
            try:
                traits[trait] = float(traits[trait])
            except (ValueError, TypeError):
                logger.error(f"Valore non valido per il tratto {trait}: {traits[trait]}")
                return jsonify({"error": f"Il valore del tratto {trait} deve essere numerico"}), 400
        
        logger.info("Avvio inferenza con i tratti forniti")
        result, model_predictions, predicted_behaviour = execute_inference_pipeline(traits, question)
        
        result = float(result)
        
        prediction_class = classify_prediction(result)
        
        response = {
            "dt_susceptibility_score": result,
            "dt_response": predicted_behaviour,
            "model_predictions": {k: float(v) if v is not None else None 
                                for k, v in model_predictions.items()},
            "classification": prediction_class
        }
        
        if 'cf' in data:
            logger.info(f"Salvataggio predizione per cf: {data['cf']}")
            prediction_record = {
                "cf": data['cf'],
                "traits": traits,
                "dt_susceptibility_score": result,
                "model_predictions": response["model_predictions"],
                "dt_response": predicted_behaviour,
                "timestamp": datetime.datetime.now()
            }
            mongodb_service.save_prediction(prediction_record)
        
        logger.info(f"Predizione completata con successo. Risultato: {result}")
        logger.info(f"Risposta completa: {json.dumps(response, indent=2)}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Errore durante la predizione: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante la predizione: {str(e)}",
            "details": "Controlla che tutti i valori dei tratti siano numerici"
        }), 500

@app.route('/extract', methods=['POST'])
def extract():
    """
    Endpoint per l'estrazione e preprocessing dei dati.
    
    Input JSON:
    {
        "process_form": true,
        "process_excel": true
    }
    """
    logger.info("Richiesta ricevuta per l'endpoint /extract")
    try:
        data = request.get_json() or {}
        
        process_form = data.get('process_form', False)
        process_excel = data.get('process_excel', True)
        
        logger.info(f"Avvio estrazione dati (form: {process_form}, excel: {process_excel})")
        processed_data = execute_extraction_preprocessing_pipeline(
            process_form=process_form,
            process_excel=process_excel
        )
        
        if not processed_data:
            logger.error("Nessun dato elaborato durante l'estrazione")
            return jsonify({"status": "warning", "message": "Nessun dato elaborato disponibile"}), 200
        
        logger.info(f"Estrazione completata con successo: {len(processed_data)} record elaborati")
        
        return jsonify({
            "status": "success",
            "message": f"Elaborati {len(processed_data)} record",
            "count": len(processed_data)
        })
    
    except Exception as e:
        logger.error(f"Errore durante l'estrazione dei dati: {e}", exc_info=True)
        return jsonify({"error": f"Errore durante l'estrazione dei dati: {str(e)}"}), 500

@app.route('/testing', methods=['POST'])
def testing():
    """
    Endpoint per eseguire i test sul modello.

    Input JSON:
    {
        "question_for_prompt": str
    }

    Returns:
        JSON: Risposta contenente le metriche di valutazione del modello:
            - classification: Metriche di classificazione (accuracy, precision, recall, f1_score)
            - regression: Metriche di regressione (mae, mse, rmse, r2_score, mape)
            
        In caso di errore, restituisce un messaggio di errore con status 500.
    """
    logger.info("Richiesta ricevuta per l'endpoint /testing")
    try:
        logger.info("Avvio esecuzione testing del modello")
        data = request.get_json()
        logger.info(f"Dati ricevuti: {json.dumps(data, indent=2)}")
        
        if 'question_for_prompt' not in data:
            logger.error("Dati mancanti nella richiesta di testing")
            return jsonify({"error": "Dati mancanti. È richiesto un JSON con il campo 'question_for_prompt'."}), 400

        question = data['question_for_prompt']

        metrics = execute_testing_pipeline(question)
        
        if not metrics:
            logger.error("Testing fallito: nessun dato disponibile")
            return jsonify({"error": "Testing fallito. Nessun dato disponibile."}), 500
        
        # Formattazione delle metriche con la nuova struttura
        formatted_metrics = {
            "classification": {
                "accuracy": float(metrics['classification']['accuracy']),
                "precision": float(metrics['classification']['precision']),
                "recall": float(metrics['classification']['recall']),
                "f1_score": float(metrics['classification']['f1_score']),
                "confusion_matrix": metrics['classification']['confusion_matrix']
            },
            "regression": {
                "mae": float(metrics['regression']['mae']),
                "mse": float(metrics['regression']['mse']),
                "rmse": float(metrics['regression']['rmse']),
                "r2_score": float(metrics['regression']['r2_score']),
                "mape": float(metrics['regression']['mape'])
            },
            "threshold_used": float(metrics['threshold_used']),
            "dataset_info": metrics.get('dataset_info', {}),
            "test_records_used": metrics.get('test_records_used', 0)
        }
        
        logger.info(f"Testing completato con successo. {json.dumps(formatted_metrics, indent=2)}")
        return jsonify(formatted_metrics)
    
    except Exception as e:
        logger.error(f"Errore durante il testing: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante il testing: {str(e)}",
            "details": "Errore nell'elaborazione delle metriche"
        }), 500
    
@app.route('/training', methods=['POST'])
def training():
    """
    Endpoint per l'addestramento dei modelli di machine learning.
    
    Questo endpoint avvia la pipeline di addestramento per i modelli KNN e DL,
    utilizzando tutti i dati disponibili in MongoDB. I modelli addestrati vengono
    salvati nella directory /models per uso futuro.
    
    Returns:
        JSON: Risposta contenente:
            - status: 'success', 'partial_success', o 'error'
            - models_trained: Lista dei modelli addestrati con successo
            - errors: Lista degli errori verificatisi durante l'addestramento
            - message: Messaggio descrittivo del risultato
            
        Status codes:
            - 200: Addestramento completato (con successo o parziale)
            - 500: Errore critico durante l'addestramento
    """
    logger.info("Richiesta ricevuta per l'endpoint /training")
    
    try:
        logger.info("Avvio esecuzione pipeline di addestramento")
        training_results = execute_training_pipeline()
        
        # Preparazione risposta
        if training_results['status'] == 'success':
            response = {
                "status": "success",
                "message": f"Addestramento completato con successo per {len(training_results['models_trained'])} modelli",
                "models_trained": training_results['models_trained'],
                "errors": []
            }
            status_code = 200
            logger.info("Pipeline di addestramento completata con successo")
            
        elif training_results['status'] == 'partial_success':
            response = {
                "status": "partial_success", 
                "message": f"Addestramento parzialmente completato. {len(training_results['models_trained'])} modelli addestrati, {len(training_results['errors'])} errori",
                "models_trained": training_results['models_trained'],
                "errors": training_results['errors']
            }
            status_code = 200
            logger.warning("Pipeline di addestramento completata parzialmente")
            
        else:  # status == 'error'
            response = {
                "status": "error",
                "message": "Addestramento fallito per tutti i modelli",
                "models_trained": [],
                "errors": training_results['errors']
            }
            status_code = 500
            logger.error("Pipeline di addestramento fallita")
        
        logger.info(f"Risposta addestramento: {json.dumps(response, indent=2)}")
        return jsonify(response), status_code
        
    except Exception as e:
        error_msg = f"Errore critico durante l'addestramento: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "models_trained": [],
            "errors": [error_msg]
        }), 500

def initialize_database():
    """
    Inizializza il database con i dati dal file Excel se è vuoto.
    """
    try:
        record_count = mongodb_service.get_record_count()
        
        if record_count == 0:
            logger.info("Database vuoto. Caricamento dati iniziali...")
            
            excel_processor = ExcelDataProcessor()
            excel_processed_data = excel_processor.process_data()
            
            if excel_processed_data:
                mongodb_service.save_data(excel_processed_data)
                logger.info(f"Database inizializzato con {len(excel_processed_data)} record.")
            else:
                logger.error("Errore: Nessun dato disponibile per l'inizializzazione del database.")
        else:
            logger.info(f"Database già inizializzato con {record_count} record.")
            
        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del database: {e}", exc_info=True)
        return False

def initialize_app():
    """
    Inizializza l'applicazione Flask.
    
    Verifica la validità della configurazione e prepara l'applicazione per l'avvio.
    
    Returns:
        bool: True se l'inizializzazione è avvenuta con successo, False altrimenti.
    """
    logger.info("Inizializzazione dell'applicazione Flask")
    try:
        if not validate_configuration():
            logger.error("Configurazione non valida. Verificare le variabili d'ambiente.")
            return False
        
        if not initialize_database():
            logger.error("Errore durante l'inizializzazione del database.")
            return False
        
        logger.info("Inizializzazione completata con successo")
        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione dell'applicazione: {e}", exc_info=True)
        return False

@app.route('/digital-twin/register', methods=['POST'])
def register_digital_twin():
    """
    Endpoint per la registrazione di un nuovo digital twin.
    
    Input JSON:
    {
        "cf": str,
        "first_name": str,
        "last_name": str,
        "traits": {
            "extraversion": float,
            "agreeableness": float,
            "conscientiousness": float,
            "neuroticism": float,
            "openness": float
        },
        "creation_datetime": str (opzionale, formato ISO),
        "last_update_datetime": str (opzionale, formato ISO),
        "last_training_datetime": str (opzionale, formato ISO)
    }
    
    Returns:
        JSON: Risultato della registrazione con ID del digital twin creato
    """
    logger.info("Richiesta ricevuta per l'endpoint /digital-twin/register")
    
    try:
        data = request.get_json()
        logger.info(f"Dati ricevuti per registrazione digital twin: {json.dumps(data, indent=2, default=str)}")
        
        if not data:
            logger.error("Dati mancanti nella richiesta di registrazione digital twin")
            return jsonify({"error": "Dati mancanti. È richiesto un JSON con i campi obbligatori."}), 400
        
        required_fields = ['cf', 'first_name', 'last_name', 'traits']
        for field in required_fields:
            if field not in data or not data[field]:
                logger.error(f"Campo obbligatorio mancante: {field}")
                return jsonify({"error": f"Campo obbligatorio mancante: {field}"}), 400
        
        traits = data['traits']
        required_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        
        for trait in required_traits:
            if trait not in traits:
                logger.error(f"Tratto mancante: {trait}")
                return jsonify({"error": f"Tratto mancante: {trait}"}), 400
            try:
                traits[trait] = float(traits[trait])
            except (ValueError, TypeError):
                logger.error(f"Valore non valido per il tratto {trait}: {traits[trait]}")
                return jsonify({"error": f"Il valore del tratto {trait} deve essere numerico"}), 400
        
        dt_data = {
            'cf': data['cf'],
            'first_name': data['first_name'],
            'last_name': data['last_name'],
            'traits': traits
        }

        for date_field in ['creation_datetime', 'last_update_datetime', 'last_training_datetime']:
            if date_field in data and data[date_field]:
                try:
                    dt_data[date_field] = datetime.datetime.fromisoformat(data[date_field].replace('Z', '+00:00'))
                except ValueError as e:
                    logger.error(f"Formato data non valido per {date_field}: {data[date_field]}")
                    return jsonify({"error": f"Formato data non valido per {date_field}. Usare formato ISO 8601."}), 400
        
        logger.info(f"Registrazione digital twin per CF: {data['cf']}")
        dt_id = mongodb_service.register_digital_twin(dt_data)
        
        if dt_id:
            response = {
                "status": "success",
                "message": f"Digital twin registrato con successo per CF: {data['cf']}",
                "digital_twin_id": dt_id,
                "cf": data['cf']
            }
            logger.info(f"Digital twin registrato con successo: {response}")
            return jsonify(response)
        else:
            logger.error("Errore durante la registrazione del digital twin")
            return jsonify({"error": "Errore durante la registrazione del digital twin"}), 500
    
    except ValueError as e:
        logger.error(f"Errore di validazione durante la registrazione: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Errore durante la registrazione del digital twin: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante la registrazione del digital twin: {str(e)}",
            "details": "Controlla che tutti i dati siano corretti e che il CF non sia già registrato"
        }), 500

@app.route('/digital-twin/predict', methods=['POST'])
def predict_by_digital_twin():
    """
    Endpoint per la predizione basata su digital twin tramite codice fiscale.
    
    Input JSON:
    {
        "cf": str
        "question_for_prompt": str
    }
    
    Returns:
        JSON: Predizione della suscettibilità al phishing, predizioni dei singoli modelli e dati del digital twin
    """
    logger.info("Richiesta ricevuta per l'endpoint /digital-twin/predict")
    
    try:
        data = request.get_json()
        logger.info(f"Dati ricevuti per predizione digital twin: {json.dumps(data, indent=2)}")
        
        if not data or 'cf' not in data:
            logger.error("Codice fiscale mancante nella richiesta di predizione digital twin")
            return jsonify({"error": "Codice fiscale mancante. È richiesto un JSON con il campo 'cf'."}), 400
        
        cf = data['cf']
        if not cf or not isinstance(cf, str):
            logger.error(f"Codice fiscale non valido: {cf}")
            return jsonify({"error": "Il codice fiscale deve essere una stringa non vuota"}), 400
        question = data['question_for_prompt']

        logger.info(f"Avvio inferenza per digital twin con CF: {cf}. Domanda effettuata: {question}")
        result, model_predictions, predicted_behaviour, digital_twin = execute_inference_pipeline_by_cf(cf, question)
        
        result = float(result)
        
        prediction_class = classify_prediction(result)
        
        response = {
            "dt_susceptibility_score": result,
            "model_predictions": {k: float(v) if v is not None else None 
                                for k, v in model_predictions.items()},
            "dt_response": predicted_behaviour,
            "digital_twin": {
                "cf": digital_twin['cf'],
                "first_name": digital_twin['first_name'],
                "last_name": digital_twin['last_name'],
                "traits": digital_twin['traits'],
                "last_training_datetime": digital_twin.get('last_training_datetime'),
                "last_update_datetime": digital_twin.get('last_update_datetime'),
                "classification": prediction_class
            }
        }
        
        prediction_record = {
            "cf": cf,
            "digital_twin_id": str(digital_twin['_id']),
            "dt_susceptibility_score": result,
            "model_predictions": response["model_predictions"],
            "dt_response": predicted_behaviour,
            "timestamp": datetime.datetime.now()
        }
        mongodb_service.save_prediction(prediction_record)
        
        logger.info(f"Predizione digital twin completata con successo per CF {cf}. Risultato: {result}")
        logger.info(f"Risposta completa: {json.dumps(response, indent=2, default=str)}")
        return jsonify(response)
    
    except ValueError as e:
        logger.error(f"Errore di validazione durante la predizione digital twin: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Errore durante la predizione digital twin: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante la predizione digital twin: {str(e)}",
            "details": "Controlla che il codice fiscale sia corretto e che il digital twin esista"
        }), 500

@app.route('/digital-twin/<cf>', methods=['GET'])
def get_digital_twin(cf):
    """
    Endpoint per recuperare i dati di un digital twin tramite codice fiscale.
    
    Returns:
        JSON: Dati del digital twin
    """
    logger.info(f"Richiesta ricevuta per l'endpoint /digital-twin/{cf}")
    
    try:
        if not cf:
            logger.error("Codice fiscale mancante nella richiesta")
            return jsonify({"error": "Codice fiscale mancante"}), 400
        
        digital_twin = mongodb_service.get_digital_twin_by_cf(cf)
        
        if not digital_twin:
            logger.warning(f"Digital twin non trovato per CF: {cf}")
            return jsonify({"error": f"Digital twin non trovato per CF: {cf}"}), 404
        
        digital_twin.pop('_id', None)
        
        logger.info(f"Digital twin recuperato con successo per CF: {cf}")
        return jsonify(digital_twin)
    
    except Exception as e:
        logger.error(f"Errore durante il recupero del digital twin per CF {cf}: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante il recupero del digital twin: {str(e)}"
        }), 500

if __name__ == "__main__":
    if initialize_app():
        logger.info("Avvio del server Flask")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Impossibile avviare l'applicazione a causa di errori di inizializzazione")
        sys.exit(1)
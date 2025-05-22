"""
Server Flask per il sistema di predizione di suscettibilità al phishing
"""
import logging
import sys
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import validate_configuration, LOG_LEVEL
from pipeline_inference.DL import DLProcessor
from pipeline_inference.llm_for_prediction import LLMPrediction
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

def execute_inference_pipeline(query_traits):
    """
    Esegue la pipeline di inferenza sui tratti del soggetto in esame.
    
    Args:
        query_traits (dict): Tratti di personalità su cui inferire
        
    Returns:
        tuple: (float, dict) - Valore finale calcolato dall'ensemble e predizioni dei singoli modelli
    """
    logger.info("Avvio pipeline di inference")

    predictions = {}
    
    # KNN
    try:
        knn_processor = KNNProcessor()
        data = mongodb_service.get_all_records()

        # Addestra il modello e trova i vicini
        knn_processor.fit(data)
        knn_result = knn_processor.process(query_traits)
        
        logger.info(f"KNN result: {knn_result}")
        predictions['KNN'] = knn_result
    except Exception as e:
        logger.error(f"Errore durante l'inferenza KNN: {e}", exc_info=True)
        predictions['KNN'] = None
    
    # Regressione
    try:
        regression_processor = RegressionProcessor()
        regression_result = regression_processor.process(query_traits)
        
        predictions['Regression'] = regression_result
        logger.info(f"REGRESSION RESULT: {regression_result}")
    except Exception as e:
        logger.error(f"Errore durante il calcolo della regressione: {e}", exc_info=True)
        predictions['Regression'] = None
    
    # LLM
    try:
        llm_predictor = LLMPrediction()
        llm_predictor_result = llm_predictor.process(query_traits)
        
        predictions['LLM'] = llm_predictor_result
        logger.info(f"LLM RESULT: {llm_predictor_result}")
    except Exception as e:
        logger.error(f"Errore durante l'inferenza LLM: {e}", exc_info=True)
        predictions['LLM'] = None
    
    # DL
    try:
        dl_processor = DLProcessor()
        data = mongodb_service.get_all_records()

        # Addestra il modello
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
        return final_result, predictions
    except Exception as e:
        logger.error(f"Errore durante il calcolo dell'ensemble: {e}", exc_info=True)
        return 0, predictions
    

def execute_testing_pipeline():
    """
    Esegue la pipeline di testing utilizzando tutti i record in MongoDB.
    
    Returns:
        dict: Metriche di valutazione del modello (accuracy, precision, recall, f1_score)
    """
    logger.info("Avvio pipeline di testing")
    
    try:
        records = mongodb_service.get_all_records()
        if not records:
            logger.error("Nessun record trovato in MongoDB per il testing")
            return None
        
        logger.info(f"Recuperati {len(records)} record da MongoDB per il testing")
        
        true_values = []
        predictions = []
        
        for record in records:
            if 'personality_traits' in record:
                traits = record['personality_traits']
                criticality = record.get('criticality_index', 0)
                
                # Esegui l'inferenza per questo record
                predicted_criticality, _ = execute_inference_pipeline(traits)
                
                predictions.append(predicted_criticality)
                true_values.append(criticality)
        
        tester = Tester(name="ModelTester")
        test_data = {
            'predictions': predictions,
            'true_values': true_values
        }
        
        metrics = tester.process(test_data)
        
        logger.info(f"Testing completato. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Errore durante il testing: {e}", exc_info=True)
        return None

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
            "/predict": "POST - Predice la suscettibilità al phishing",
            "/extract": "POST - Esegue estrazione e preprocessing dei dati",
            "/test": "GET - Esegue test sul modello",
            "/health": "GET - Verifica lo stato del servizio"
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
        "user_id": str (opzionale)
    }
    
    Returns:
        JSON: Predizione della suscettibilità al phishing e predizioni dei singoli modelli
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
        
        # Verifica e converti i tratti in float
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
        result, model_predictions = execute_inference_pipeline(traits)
        
        # Assicurati che result sia float
        result = float(result)
        
        response = {
            "prediction": result,
            "susceptibility_score": result,
            "model_predictions": {k: float(v) if v is not None else None 
                                for k, v in model_predictions.items()}
        }
        
        if 'user_id' in data:
            logger.info(f"Salvataggio predizione per user_id: {data['user_id']}")
            prediction_record = {
                "user_id": data['user_id'],
                "traits": traits,
                "prediction": result,
                "model_predictions": response["model_predictions"],
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

@app.route('/testing', methods=['GET'])
def testing():
    """
    Endpoint per eseguire i test sul modello.
    
    Returns:
        JSON: Risposta contenente le metriche di valutazione del modello:
            - accuracy: Accuratezza del modello
            - precision: Precisione del modello
            - recall: Recall del modello
            - f1_score: F1 score del modello
            
        In caso di errore, restituisce un messaggio di errore con status 500.
    """
    logger.info("Richiesta ricevuta per l'endpoint /testing")
    try:
        logger.info("Avvio esecuzione testing del modello")
        metrics = execute_testing_pipeline()
        
        if not metrics:
            logger.error("Testing fallito: nessun dato disponibile")
            return jsonify({"error": "Testing fallito. Nessun dato disponibile."}), 500
        
        # Converti tutti i valori numerici in float e gestisci la matrice di confusione
        formatted_metrics = {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1_score": float(metrics['f1_score']),
        }
        
        logger.info(f"Testing completato con successo. {json.dumps(formatted_metrics, indent=2)}")
        return jsonify(formatted_metrics)
    
    except Exception as e:
        logger.error(f"Errore durante il testing: {e}", exc_info=True)
        return jsonify({
            "error": f"Errore durante il testing: {str(e)}",
            "details": "Errore nell'elaborazione delle metriche"
        }), 500
    
@app.route('/training', methods=['GET'])
def training():
    return True
    #TODO implementare

def initialize_database():
    """
    Inizializza il database con i dati dal file Excel se è vuoto.
    """
    try:
        # Verifica se il database è vuoto
        record_count = mongodb_service.get_record_count()
        
        if record_count == 0:
            logger.info("Database vuoto. Caricamento dati iniziali...")
            
            # Carica i dati dal file Excel
            excel_processor = ExcelDataProcessor()
            excel_processed_data = excel_processor.process_data()
            
            if excel_processed_data:
                # Salva i dati in MongoDB
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
        
        # Inizializza il database se necessario
        if not initialize_database():
            logger.error("Errore durante l'inizializzazione del database.")
            return False
        
        logger.info("Inizializzazione completata con successo")
        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione dell'applicazione: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if initialize_app():
        logger.info("Avvio del server Flask")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Impossibile avviare l'applicazione a causa di errori di inizializzazione")
        sys.exit(1)
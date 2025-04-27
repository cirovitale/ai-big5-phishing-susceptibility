"""
Main
"""
import logging
import sys
from config import validate_configuration, LOG_LEVEL
from services.google_service import GoogleService
from services.mongodb_service import MongoDBService
from pipeline_preprocessing.survey_form import SurveyDataProcessor
from pipeline_preprocessing.dataset_excel import ExcelDataProcessor
from pipeline_inference.knn import KNNProcessor
from pipeline_inference.regression import RegressionProcessor
from pipeline_inference.ensemble import EnsembleProcessor
from pipeline_testing.tester import Tester

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('app.log')]
)

logger = logging.getLogger(__name__)

# Inizializzazione servizi
google_service = GoogleService()
mongodb_service = MongoDBService()

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
            logger.info("Elaborazione dati dal form Google")
            form_processor = SurveyDataProcessor()
            
            # Autenticazione e recupero dati
            form_data = google_service.get_form_responses()
            
            # Elaborazione dati form
            form_processed_data = form_processor.process_data(form_data)
            all_processed_data.extend(form_processed_data)
            logger.info(f"Aggiunti {len(form_processed_data)} record dal form Google")
        
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
        query_traits (dict): (KNN) Tratti di personalità da cercare
        n_neighbors (int): (KNN) Numero di vicini da trovare
        
    Returns:
        float: Valore finale calcolato dall'ensemble
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
        # TODO
        predictions['LLM'] = 0
    except Exception as e:
        logger.error(f"Errore durante l'inferenza LLM: {e}", exc_info=True)
        predictions['LLM'] = None
    
    # DL
    try:
        # TODO
        predictions['DL'] = 0
    except Exception as e:
        logger.error(f"Errore durante l'inferenza DL: {e}", exc_info=True)
        predictions['DL'] = None
    
    # LSTM
    try:
        # TODO
        predictions['LSTM'] = 0
    except Exception as e:
        logger.error(f"Errore durante l'inferenza LSTM: {e}", exc_info=True)
        predictions['LSTM'] = None
    
    # ENSEMBLE
    try:
        ensemble_processor = EnsembleProcessor()
        final_result = ensemble_processor.process(predictions)
        
        logger.info(f"ENSEMBLE RESULT: {final_result}")
        return final_result
    except Exception as e:
        logger.error(f"Errore durante il calcolo dell'ensemble: {e}", exc_info=True)
        return 0
    

def execute_testing_pipeline():
    """
    Esegue la pipeline di testing utilizzando tutti i record in MongoDB.
    
    Returns:
        dict: Metriche di valutazione
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
                predicted_criticality = execute_inference_pipeline(traits)
                
                predictions.append(predicted_criticality)
                true_values.append(criticality)
        
        tester = Tester(name="ModelTester")
        test_data = {
            'predictions': predictions,
            'true_values': true_values
        }
        
        metrics = tester.process(
            test_data
        )
        
        logger.info(f"Testing completato. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Errore durante il testing: {e}", exc_info=True)
        return None
    
def main():
    """
    Main
    """
    try:
        # Validazione configurazione
        if not validate_configuration():
            logger.error("Configurazione non valida. Verificare le variabili d'ambiente.")
            return False
        
        logger.info("Avvio dell'applicazione")
        
        # Configurazione modalità di esecuzione
        execute_mode = "testing"  # "extraction", "inference", "testing", "training" (TODO)
        
        if execute_mode == "extraction":
            processed_data = execute_extraction_preprocessing_pipeline(process_excel=True)
            if not processed_data:
                logger.error("Nessun dato elaborato disponibile")
                return False

        elif execute_mode == "testing":
            metrics = execute_testing_pipeline()
            
            if metrics:
                print("\n=== Risultati Testing ===")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                
                # Stampa la matrice di confusione
                print("\nConfusion Matrix:")
                cm = metrics['confusion_matrix']
                print(f"                Predicted:")
                print(f"                Non-Critical  Critical")
                print(f"Actual: Non-Critical  {cm[0][0]:4d}         {cm[0][1]:4d}")
                print(f"        Critical      {cm[1][0]:4d}         {cm[1][1]:4d}")
            # else:
            #     print("Testing fallito.")

        elif execute_mode == "inference":
            # TODO
            # Tratti in Input
            query_traits = {
                'extraversion': 0.5,
                'agreeableness': 0.4,
                'conscientiousness': 0.3,
                'neuroticism': 0.8,
                'openness': 0.1
            }
            
            # Inferenza
            result = execute_inference_pipeline(query_traits)
            print(f"Risultato dell'inferenza: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione dell'applicazione: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
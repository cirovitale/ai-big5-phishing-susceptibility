"""
Modulo per l'elaborazione dei dati provenienti da file Excel.

Questo modulo fornisce funzionalità per analizzare e trasformare
dati provenienti da dataset in Excel, eseguendo calcoli e preparandoli
per l'inserimento in MongoDB.
"""

import logging
import os
import pandas as pd
import numpy as np  # Aggiunto per il random splitting
from datetime import datetime
from pipeline_preprocessing.data_processor_base import DataProcessorBase
from config import DATASET_PATH, SCALING_TYPE

logger = logging.getLogger(__name__)

class ExcelDataProcessor(DataProcessorBase):
    """
    Classe per l'elaborazione dei dati provenienti da file Excel.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Inizializza il data processor di dati Excel.
        
        Args:
            test_size (float): Percentuale di dati da riservare per il testing (default: 0.2 = 20%)
            random_state (int): Seed per la riproducibilità della divisione train/test
        """
        super().__init__()
        self.source_data = DATASET_PATH
        self.scaling_type = SCALING_TYPE
        self.test_size = test_size
        self.random_state = random_state

        # Statistiche del dataset
        self.dataset_stats = {
            'delta': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'std': 0.0
            },
            'ocean': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'std': 0.0
            }
        }

        # Definizione colonne
        self.PERSUASION_COLUMNS = [
            'Social_Proof_Delta_Install',
            'Likeability_Delta_Install',
            'Authority_Delta_Install',
            'Commitment_Consistence_Delta_Install',
            'Reciprocity_Delta_Install',
            'Scarcity_Delta_Install'
        ]
        
        self.OCEAN_COLUMNS = [
            'Extraversion', 
            'Agreeableness', 
            'Conscientiousness', 
            'Neuroticism', 
            'Openness'
        ]
        
        self.EXTRACTION_COLUMNS = [
            'Group', 'Age', 'Gender'
        ] + self.OCEAN_COLUMNS + self.PERSUASION_COLUMNS
            
        logger.info(f"Inizializzato data processor Excel con scaling {SCALING_TYPE}")
    
    def scale_value(self, value, feature_type='ocean'):
        """
        Applica lo scaling specificato al valore.
        
        Args:
            value (float): Valore da scalare
            feature_type (str): Tipo di feature ('ocean' o 'delta')
            
        Returns:
            float: Valore scalato
        """
        try:
            if pd.isna(value):
                return 0.0
                
            value = float(value)
            dataset_stats = self.dataset_stats[feature_type]

            if self.scaling_type == "min_max":
                min_val = dataset_stats['min']
                max_val = dataset_stats['max']
                if max_val == min_val:
                    return 0.0
                return (value - min_val) / (max_val - min_val)
            
            elif self.scaling_type == "standard":
                mean = dataset_stats['mean']
                std = dataset_stats['std']
                if std == 0:
                    return 0.0
                return (value - mean) / std
            
            else:
                logger.warning(f"Tipo di scaling non supportato: {self.scaling_type}")
                return value
                
        except Exception as e:
            logger.error(f"Errore durante lo scaling del valore {value}: {e}")
            return 0.0

    def calculate_dataset_statistics(self, df):
        """
        Calcola le statistiche del dataset necessarie per lo scaling.
        
        Args:
            df (pandas.DataFrame): DataFrame con i dati
        """
        try:
            # Statistiche per le colonne di persuasione
            persuasion_data = df[self.PERSUASION_COLUMNS].values.flatten()
            persuasion_data = persuasion_data[~pd.isna(persuasion_data)]
            
            self.dataset_stats['delta'].update({
                'min': float(persuasion_data.min()),
                'max': float(persuasion_data.max()),
                'mean': float(persuasion_data.mean()),
                'std': float(persuasion_data.std())
            })
            
            # Statistiche per le colonne OCEAN
            ocean_data = df[self.OCEAN_COLUMNS].values.flatten()
            ocean_data = ocean_data[~pd.isna(ocean_data)]
            
            self.dataset_stats['ocean'].update({
                'min': float(ocean_data.min()),
                'max': float(ocean_data.max()),
                'mean': float(ocean_data.mean()),
                'std': float(ocean_data.std())
            })
            
            logger.info("Statistiche del dataset calcolate con successo")
            logger.debug(f"Statistiche: {self.dataset_stats}")
            
        except Exception as e:
            logger.error(f"Errore nel calcolo delle statistiche del dataset: {e}")
            raise

    def calculate_criticality_index(self, row):
        """
        Calcola l'indice di criticità per una riga di dati.
        
        Args:
            row (pandas.Series): Riga di dati dal DataFrame.
            
        Returns:
            float: Indice di criticità calcolato.
        """
        try:
            values = []
            for column in self.PERSUASION_COLUMNS:
                if column in row and pd.notna(row[column]):
                    values.append(float(row[column]))
            
            if not values:
                return 0.0
            
            # Media dei valori prima dello scaling
            raw_index = sum(values) / len(values)
            
            # Applica lo scaling all'indice di criticità
            scaled_index = self.scale_value(raw_index, 'delta')
            
            return float(round(scaled_index, 4))
            
        except Exception as e:
            logger.error(f"Errore nel calcolo dell'indice di criticità: {e}")
            return 0.0
    
    def read_excel(self):
        """
        Legge il file Excel (dataset).

        Returns:
            df (pandas.DataFrame): DataFrame contenente i dati letti dal file Excel.
        """
        try:
            if isinstance(self.source_data, str):
                if not os.path.exists(self.source_data):
                    raise FileNotFoundError(f"File Excel non trovato: {self.source_data}")
                
                if os.path.isdir(self.source_data):
                    excel_files = [f for f in os.listdir(self.source_data) if f.endswith('.xlsx')]
                    if not excel_files:
                        raise FileNotFoundError(f"Nessun file Excel trovato nella directory: {self.source_data}")
                    
                    # Usa il primo file Excel trovato
                    file_path = os.path.join(self.source_data, excel_files[0])
                    logger.info(f"Utilizzando il file Excel trovato nella directory: {file_path}")
                else:
                    file_path = self.source_data
                
                df = pd.read_excel(file_path)
                logger.info(f"Caricato file Excel: {file_path} con {len(df)} righe")
                return df
            else:
                raise ValueError("source_data deve essere un percorso file di un dataset")
        except Exception as e:
            logger.error(f"Errore nella lettura del file Excel: {self.source_data}, errore: {e}")
            raise

    def build_records(self, df):
        """
        Costruisce i record elaborati a partire dal DataFrame con divisione train/test.

        Args:
            df (pandas.DataFrame): DataFrame contenente i dati letti dal file Excel.

        Returns:
            list: Lista di record elaborati pronti per l'inserimento in MongoDB.
        """
        processed_data = []
        # df = df[df['Group'] != 'Arab']
        
        # Divisione stratificata train/test basata sui dati
        np.random.seed(self.random_state)
        total_records = len(df)
        test_indices = np.random.choice(
            total_records, 
            size=int(total_records * self.test_size), 
            replace=False
        )
        
        logger.info(f"Divisione dataset: {total_records} record totali, {len(test_indices)} per testing ({self.test_size*100:.1f}%), {total_records - len(test_indices)} per training ({(1-self.test_size)*100:.1f}%)")

        for idx, row in df.iterrows():
            try:
                # Determina se questo record è per il testing
                is_test_record = idx in test_indices
                
                record = {
                    'source': 'excel_dataset',
                    'validation': is_test_record,
                    'demographic_traits': {
                        'country': row.get('Group', ''),
                        'age': int(row.get('Age', 0)) if pd.notna(row.get('Age', 0)) else 0,
                        'gender': row.get('Gender', ''),
                    },
                    'personality_traits': {
                        trait.lower(): self.scale_value(
                            row.get(trait, 0), 
                            'ocean'
                        ) for trait in self.OCEAN_COLUMNS
                    },
                    'criticality_index': self.calculate_criticality_index(row),
                    'survey_raw_data': {
                        col: float(row.get(col, 0)) if pd.notna(row.get(col, 0)) else 0
                        for col in self.PERSUASION_COLUMNS
                    }
                }
                processed_data.append(record)
            except Exception as e:
                logger.error(f"Errore nell'elaborazione della riga {idx}: {e}")
                continue

        # Log della divisione finale
        train_count = sum(1 for record in processed_data if not record['validation'])
        test_count = sum(1 for record in processed_data if record['validation'])
        logger.info(f"Record elaborati: {train_count} training, {test_count} testing, {len(processed_data)} totali")
        
        return processed_data

    def process_data(self):
        """
        Elabora i dati dal file Excel e calcola l'indice di criticità.

        Returns:
            list: Lista di record elaborati e normalizzati.
        """
        try:
            df = self.read_excel()
            self.calculate_dataset_statistics(df)
            processed_data = self.build_records(df)
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione del file Excel: {e}", exc_info=True)
            raise
        return processed_data
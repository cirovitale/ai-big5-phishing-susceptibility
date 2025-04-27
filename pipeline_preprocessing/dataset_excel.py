"""
Modulo per l'elaborazione dei dati provenienti da file Excel.

Questo modulo fornisce funzionalità per analizzare e trasformare
dati provenienti da dataset in Excel, eseguendo calcoli e preparandoli
per l'inserimento in MongoDB.
"""

import logging
import os
import pandas as pd
from datetime import datetime
from pipeline_preprocessing.data_processor_base import DataProcessorBase
from config import DATASET_PATH

logger = logging.getLogger(__name__)


class ExcelDataProcessor(DataProcessorBase):
    """
    Classe per l'elaborazione dei dati provenienti da file Excel.
    """
    
    def __init__(self):
        """
        Inizializza il data processor di dati Excel.
        """
        super().__init__()

        self.source_data = DATASET_PATH

        # Definizione delle colonne per il calcolo dell'indice di criticità
        self.PERSUASION_COLUMNS = [
            'Social_Proof_Delta_Install',
            'Likeability_Delta_Install',
            'Authority_Delta_Install',
            'Commitment_Consistence_Delta_Install',
            'Reciprocity_Delta_Install',
            'Scarcity_Delta_Install'
        ]
        
        # Definizione delle colonne di base per l'estrazione
        self.EXTRACTION_COLUMNS = [
            'Group', 'Age', 'Gender',
            'Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'
        ] + self.PERSUASION_COLUMNS        

        self.dataset_delta_min = float('inf')
        self.dataset_delta_max = float('-inf')

        self.dataset_ocean_min = float('inf')
        self.dataset_ocean_max = float('-inf')
            
        logger.info(f"Inizializzato data processor Excel")
    
    def calculate_criticality_index(self, row):
        """
        Calcola l'indice di criticità per una riga di dati.
        
        Args:
            row (pandas.Series): Riga di dati dal DataFrame.
            
        Returns:
            float: Indice di criticità calcolato.
        """
        sum_crit_columns = 0.0
        n_crit_columns = 0
        
        for column in self.PERSUASION_COLUMNS:
            if column in row and pd.notna(row[column]):
                sum_crit_columns += float(row[column])
                n_crit_columns += 1
        
        criticality_index = sum_crit_columns / n_crit_columns if n_crit_columns > 0 else 0

        normalized_index = self.min_max_scaling(
            value=criticality_index, 
            min_val=self.dataset_delta_min, 
            max_val=self.dataset_delta_max
        )
        
        return float(round(normalized_index, 2))
    
    def min_max_scaling(self, value, min_val, max_val):
        """
        Applica il min-max scaling per normalizzare un valore in un intervallo target.
        
        Args:
            value (float): Il valore da normalizzare
            min_val (float): Il valore minimo nell'intervallo originale
            max_val (float): Il valore massimo nell'intervallo originale
            
        Returns:
            float: Il valore normalizzato nell'intervallo target
        """
        normalized = (value - min_val) / (max_val - min_val)
    
        return normalized
    
    def read_excel(self):
        """
        Legge il file Excel (dataset).

        Returns:
            df (pandas.DataFrame): DataFrame contenente i dati letti dal file Excel.
        """
        if isinstance(self.source_data, str):
            if not os.path.exists(self.source_data):
                raise FileNotFoundError(f"File Excel non trovato: {self.source_data}")
            df = pd.read_excel(self.source_data)
            logger.info(f"Caricato file Excel: {self.source_data} con {len(df)} righe")
            return df
        else:
            raise ValueError("source_data deve essere un percorso file di un dataset")

    def calculate_min_max(self, df):
        """
        Calcola i valori minimi e massimi delle colonne di persuasione e dei tratti di personalità.

        Args:
            df (pandas.DataFrame): DataFrame contenente i dati letti dal file Excel.
        """
        OCEAN_COLUMNS = [
            'Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'
        ]

        for col in self.PERSUASION_COLUMNS:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_min) and col_min < self.dataset_delta_min:
                    self.dataset_delta_min = col_min
                if pd.notna(col_max) and col_max > self.dataset_delta_max:
                    self.dataset_delta_max = col_max

        for col in OCEAN_COLUMNS:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_min) and col_min < self.dataset_ocean_min:
                    self.dataset_ocean_min = col_min
                if pd.notna(col_max) and col_max > self.dataset_ocean_max:
                    self.dataset_ocean_max = col_max

    def build_records(self, df):
        """
        Costruisce i record elaborati a partire dal DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame contenente i dati letti dal file Excel.

        Returns:
            list: Lista di record elaborati pronti per l'inserimento in MongoDB.
        """
        processed_data = []
        # Filtra le righe per Group (country) diverso da 'Arab'
        df = df[df['Group'] != 'Arab']

        for _, row in df.iterrows():
            record = {
                'source': 'excel_dataset',
                'demographic_traits': {
                    'country': row.get('Group', ''),
                    'age': int(row.get('Age', 0)) if pd.notna(row.get('Age', 0)) else 0,
                    'gender': row.get('Gender', ''),
                },
                'personality_traits': {
                    'extraversion': self.min_max_scaling(
                        value=float(row.get('Extraversion', 0)) if pd.notna(row.get('Extraversion', 0)) else 0,
                        min_val=self.dataset_ocean_min,
                        max_val=self.dataset_ocean_max
                    ),
                    'agreeableness': self.min_max_scaling(
                        value=float(row.get('Agreeableness', 0)) if pd.notna(row.get('Agreeableness', 0)) else 0,
                        min_val=self.dataset_ocean_min,
                        max_val=self.dataset_ocean_max
                    ),
                    'conscientiousness': self.min_max_scaling(
                        value=float(row.get('Conscientiousness', 0)) if pd.notna(row.get('Conscientiousness', 0)) else 0,
                        min_val=self.dataset_ocean_min,
                        max_val=self.dataset_ocean_max
                    ),
                    'neuroticism': self.min_max_scaling(
                        value=float(row.get('Neuroticism', 0)) if pd.notna(row.get('Neuroticism', 0)) else 0,
                        min_val=self.dataset_ocean_min,
                        max_val=self.dataset_ocean_max
                    ),
                    'openness': self.min_max_scaling(
                        value=float(row.get('Openness', 0)) if pd.notna(row.get('Openness', 0)) else 0,
                        min_val=self.dataset_ocean_min,
                        max_val=self.dataset_ocean_max
                    ),
                },
                'criticality_index': float(self.calculate_criticality_index(row)),
                'survey_raw_data': {
                    col: float(row.get(col, 0)) if pd.notna(row.get(col, 0)) else 0
                    for col in self.PERSUASION_COLUMNS
                }
            }
            processed_data.append(record)
        logger.info(f"Elaborati {len(processed_data)} record dal file Excel")
        return processed_data

    def process_data(self):
        """
        Elabora i dati dal file Excel e calcola l'indice di criticità.

        Returns:
            list: Lista di record elaborati e normalizzati.
        """
        try:
            df = self.read_excel()
            self.calculate_min_max(df)
            processed_data = self.build_records(df)
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione del file Excel: {e}", exc_info=True)
            raise
        return processed_data
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from services.mongodb_service import MongoDBService
from config import LOG_LEVEL
import os

# Crea la cartella 'chart' se non esiste
CHART_DIR = "chart"
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('app.log')]
)

logger = logging.getLogger(__name__)

CRITICALITY_THRESHOLD = float(os.getenv("CRITICALITY_THRESHOLD", 0.5))

def extract_personality_traits(documents):
    """
    Estrae solo i tratti di personalità OCEAN e l'indice di criticità dai documenti.

    Args:
        documents (list): Lista di documenti provenienti dal database MongoDB.

    Returns:
        pandas.DataFrame: DataFrame contenente i tratti OCEAN e l'indice di criticità per ciascun documento.
    """
    personality_data = []
    
    for doc in documents:
        personality = doc.get('personality_traits', {})
        
        data = {
            'extraversion': float(personality.get('extraversion', 0)),
            'agreeableness': float(personality.get('agreeableness', 0)),
            'conscientiousness': float(personality.get('conscientiousness', 0)),
            'neuroticism': float(personality.get('neuroticism', 0)),
            'openness': float(personality.get('openness', 0)),
            'criticality_index': float(doc.get('criticality_index', 0))
        }
        
        personality_data.append(data)
    
    return pd.DataFrame(personality_data)

def plot_personality_traits_susceptibility(df):
    """
    Crea un barplot per visualizzare la correlazione tra i tratti di personalità OCEAN e la suscettibilità al phishing.

    Args:
        df (pandas.DataFrame): DataFrame contenente i tratti OCEAN e l'indice di criticità.
    """
    big_five_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    
    correlations = [df[trait].corr(df['criticality_index']) for trait in big_five_traits]
    
    correlation_df = pd.DataFrame({
        'trait': [t.capitalize() for t in big_five_traits],
        'correlation': correlations
    })
    
    correlation_df = correlation_df.sort_values('correlation', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(correlation_df['trait'], correlation_df['correlation'])
    
    for i, bar in enumerate(bars):
        if correlation_df['correlation'].iloc[i] >= 0:
            bar.set_color('steelblue')
        else:
            bar.set_color('firebrick')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlazione tra Tratti di Personalità e Suscettibilità al Phishing')
    plt.ylabel('Coefficiente di Correlazione')
    plt.xlabel('Tratti di Personalità (Big Five/OCEAN)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    for i, v in enumerate(correlation_df['correlation']):
        plt.text(i, v + (0.02 if v >= 0 else -0.05), 
                 f'{v:.2f}', 
                 ha='center', 
                 fontweight='bold')
    
    plt.savefig(os.path.join(CHART_DIR, 'personality_phishing_correlation.png'))
    plt.show()

def calculate_pearson_correlations(df):
    """
    Calcola e stampa le correlazioni di Pearson tra i tratti OCEAN e l'indice di criticità.

    Args:
        df (pandas.DataFrame): DataFrame contenente i tratti OCEAN e l'indice di criticità.
    """
    big_five_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    
    logging.info("Correlazioni di Pearson con l'indice di criticità:")
    logging.info("="*50)
    
    for trait in big_five_traits:
        correlation, p_value = stats.pearsonr(df[trait], df['criticality_index'])
        
        significance = "significativa" if p_value < 0.05 else "non significativa"
        
        logging.info(f"{trait.capitalize()}: r = {correlation:.4f}, p-value = {p_value:.4f} ({significance})")
    
    logging.info("="*50)
    
    return None

def print_dataframe_head(df, n=5):
    """
    Mostra le prime righe del DataFrame tramite logging.

    Args:
        df (pandas.DataFrame): DataFrame da visualizzare.
        n (int): Numero di righe da mostrare.
    """
    logging.info("\nPrime righe del DataFrame:")
    logging.info(f"\n{df.head(n)}")

def print_descriptive_statistics(df):
    """
    Mostra le statistiche descrittive del DataFrame tramite logging.

    Args:
        df (pandas.DataFrame): DataFrame da analizzare.
    """
    logging.info("\nStatistiche descrittive:")
    logging.info(f"\n{df.describe()}")

def plot_correlation_heatmap(df):
    """
    Crea e salva una heatmap della matrice di correlazione.

    Args:
        df (pandas.DataFrame): DataFrame da analizzare.
    """
    correlation_matrix = df.corr()
    logging.info("\nCorrelazioni con l'indice di criticità:")
    logging.info(f"\n{correlation_matrix['criticality_index'].sort_values(ascending=False)}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice di Correlazione tra Tratti OCEAN e Suscettibilità')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'ocean_correlation_heatmap.png'))
    plt.show()

def plot_susceptibility_counts(df, threshold=CRITICALITY_THRESHOLD):
    """
    Conta e visualizza il numero di soggetti suscettibili e non suscettibili.

    Args:
        df (pandas.DataFrame): DataFrame con l'indice di criticità.
        threshold (float): Soglia per definire la suscettibilità.
    """
    df['susceptible'] = df['criticality_index'] >= threshold
    counts = df['susceptible'].value_counts().sort_index()
    labels = ['Non suscettibili', 'Suscettibili']
    values = [counts.get(False, 0), counts.get(True, 0)]

    logging.info(f"Numero di non suscettibili (indice < {threshold}): {values[0]}")
    logging.info(f"Numero di suscettibili (indice >= {threshold}): {values[1]}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['steelblue', 'firebrick'])
    plt.ylabel('Numero di soggetti')
    plt.title(f'Conteggio soggetti suscettibili (soglia = {threshold})')
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'susceptibility_counts.png'))
    plt.show()

def plot_ocean_boxplots_by_susceptibility(df, threshold=CRITICALITY_THRESHOLD):
    """
    Crea boxplot dei tratti OCEAN separati per suscettibili e non suscettibili.

    Args:
        df (pandas.DataFrame): DataFrame con tratti OCEAN e suscettibilità.
        threshold (float): Soglia per la suscettibilità.
    """
    df['susceptible'] = df['criticality_index'] >= threshold
    melted = df.melt(id_vars='susceptible', value_vars=['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness'],
                     var_name='Trait', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Trait', y='Value', hue='susceptible', data=melted, palette=['steelblue', 'firebrick'])
    plt.title('Distribuzione dei tratti OCEAN per suscettibili e non suscettibili')
    plt.ylabel('Valore normalizzato')
    plt.xlabel('Tratto di personalità')
    plt.legend(title='Suscettibile', labels=['No', 'Sì'])
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'ocean_boxplot_susceptibility.png'))
    plt.show()

def main():
    """
    Esegue l'analisi completa dei dati del dataset di personalità dopo la normalizzazione.
    """
    mongodb_service = MongoDBService()
    mongodb_service.connect()
    documents = mongodb_service.get_all_dataset_records()
    if not documents:
        logging.warning("Nessun documento del dataset trovato nel database.")
        return
    logging.info(f"Trovati {len(documents)} documenti del dataset.")
    df = extract_personality_traits(documents)
    print_dataframe_head(df)
    print_descriptive_statistics(df)
    plot_personality_traits_susceptibility(df)
    plot_correlation_heatmap(df)
    calculate_pearson_correlations(df)
    plot_susceptibility_counts(df, threshold=CRITICALITY_THRESHOLD)
    plot_ocean_boxplots_by_susceptibility(df, threshold=CRITICALITY_THRESHOLD)

if __name__ == "__main__":
    main()
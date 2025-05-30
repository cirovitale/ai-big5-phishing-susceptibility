# Language:

- 🇬🇧 [ENGLISH VERSION](#english-version)
- 🇮🇹 [ITALIAN VERSION](#italian-version)

# ENGLISH VERSION

# AI-BIG5-Phishing-Susceptibility

Initially designed for the University of Salerno's Artificial Intelligence course, then engineered according to the principles of Software Engineering for Artificial Intelligence, explored in depth in the University of Salerno's course of the same name, for industrial deployment.

## 👥 Team

- [Simone Scermino](https://github.com/Hikki00)
- [Ciro Vitale](https://github.com/cirovitale)

## 📝 Description

A flexible, production-ready system to predict user susceptibility to phishing attacks based on the Big Five (OCEAN) personality traits. Born from academic research (literature review on Big Five correlations and demographic factors), our solution implements and ensembles four complementary models: KNN, Regression, Deep Learning, and an LLM-based approach—to maximize accuracy and robustness.

## 🛠️ Technologies

- **Backend**: Python, Flask
- **Models**: scikit-learn, TensorFlow/Keras, OpenAI API
- **Database**: MongoDB
- **Explainability**: SHAP
- **Deployment**: Docker, Docker Compose

## Server

### 1. Clone repository

```bash
git clone https://github.com/cirovitale/ai-big5-phishing-susceptibility.git
cd ai-big5-phishing-susceptibility
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```dotenv
# OpenAI API
OPENAI_API_KEY=<your_secret_key>

# MongoDB
MONGODB_URI=mongodb://userAdmin:password@mongodb:27017/ai-bi5-phishing-susceptibility?authSource=admin
MONGODB_URI_USER=userAdmin
MONGODB_URI_PASSWORD=password
MONGODB_DB=ai-bi5-phishing-susceptibility
MONGODB_COLLECTION_INFERENCE=prediction
MONGODB_COLLECTION_DATASET=dataset
MONGODB_COLLECTION_DT=digital-twin

# Dataset Processing
DATASET_PATH = dataset/Dataset.xlsx
CRITICALITY_THRESHOLD = 0.5
SCALING_TYPE = "min_max" # o "standard"

# Logging
LOG_LEVEL=INFO

# Ensemble Weights Configuration
ENSEMBLE_WEIGHT_KNN=1
ENSEMBLE_WEIGHT_REGRESSION=1
ENSEMBLE_WEIGHT_LLM=1
ENSEMBLE_WEIGHT_DL=1
```

### 3. Build and launch with Docker Compose

```bash
docker-compose build --no-cache
docker-compose up -d
```

This will start:

- **app** on port `5000`
- **mongo** on port `27017`

### 4. Access Services

- **API Docs**: [http://localhost:5000/](http://localhost:5000/)

## Client

### 1. Setup client environment

Create and activate a virtual environment:

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install client dependencies

```bash
pip install -r requirements-client.txt
```

### 3. Run API tests

Ensure the server is running, then execute the test client:

```bash
python client.py
```

### 5. Teardown

```bash
docker-compose down
```

---

# ITALIAN VERSION

# AI-BIG5-Phishing-Susceptibility

Inizialmente ideato per il corso di Intelligenza Artificiale dell'Università degli Studi di Salerno, poi ingegnerizzato secondo i principi di Software Engineering for Artificial Intelligence, approfonditi nell'omonimo corso dell'Università degli Studi di Salerno, per un deploy industriale.

## 👥 Team

- [Simone Scermino](https://github.com/Hikki00)
- [Ciro Vitale](https://github.com/cirovitale)

## 📝 Descrizione

Un sistema flessibile e pronto per la messa in produzione per prevedere la suscettibilità degli utenti agli attacchi di phishing sulla base dei tratti di personalità dei Big Five (OCEAN). Supportato da una revisione della letteratura sulle correlazioni dei Big Five, la nostra soluzione implementa ed effettua l'ensemble di quattro modelli complementari: KNN, Regressione, Deep Learning e un approccio basato su LLM per massimizzare l'accuratezza e la robustezza.

## 🛠️ Tecnologie

- **Backend**: Python, Flask
- **Modelli**: scikit-learn, TensorFlow/Keras, API OpenAI
- **Database**: MongoDB
- **Explainability**: SHAP
- **Deployment**: Docker, Docker Compose

## Server

### 1. Clona il repository

```bash
git clone https://github.com/cirovitale/ai-big5-phishing-susceptibility.git
cd ai-big5-phishing-susceptibility
```

### 2. Configurazione ambiente

```bash
cp .env.example .env
```

Modifica `.env` con i tuoi valori:

```dotenv
# OpenAI API
OPENAI_API_KEY=<your_secret_key>

# MongoDB
MONGODB_URI=mongodb://userAdmin:password@mongodb:27017/ai-bi5-phishing-susceptibility?authSource=admin
MONGODB_URI_USER=userAdmin
MONGODB_URI_PASSWORD=password
MONGODB_DB=ai-bi5-phishing-susceptibility
MONGODB_COLLECTION_INFERENCE=prediction
MONGODB_COLLECTION_DATASET=dataset
MONGODB_COLLECTION_DT=digital-twin

# Dataset Processing
DATASET_PATH = dataset/Dataset.xlsx
CRITICALITY_THRESHOLD = 0.5
SCALING_TYPE = "min_max" # o "standard"

# Logging
LOG_LEVEL=INFO

# Ensemble Weights Configuration
ENSEMBLE_WEIGHT_KNN=1
ENSEMBLE_WEIGHT_REGRESSION=1
ENSEMBLE_WEIGHT_LLM=1
ENSEMBLE_WEIGHT_DL=1
```

### 3. Builda e avvia con Docker Compose

```bash
docker-compose build --no-cache
docker-compose up -d
```

Si avvieranno:

- **app** su porta `5000`
- **mongo** su porta `27017`

### 4. Accedi ai servizi

- **Documentazione API**: [http://localhost:5000/](http://localhost:5000/)

## Client

### 1. Configura l'ambiente client

Crea e attiva un ambiente virtuale:

```bash
python -m venv venv

# Su Windows:
venv\Scripts\activate
# Su macOS/Linux:
source venv/bin/activate
```

### 2. Installa le dipendenze del client

```bash
pip install -r requirements-client.txt
```

### 3. Esegui i test delle API

Assicurati che il server sia in esecuzione, poi esegui il client di test:

```bash
python client.py
```

### 5. Arresto

```bash
docker-compose down
```

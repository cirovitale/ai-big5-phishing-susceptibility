# ai-big5-phishing-susceptibility

<!-- MongoDB -->

installazione Docker MongoDB e config .ENV

<!-- Google Cloud -->

Generare file di autenticazione GUIDA e config .ENV

nome service account: ai-big5-phishing-susceptibility
email: ai-big5-phishing-susceptibilit@aesthetic-hub-449411-f3.iam.gserviceaccount.com
form-id: 1hE0Hbp_HJu4yrnOppYivRx6d--s0cfekIl_c7mjIIl0

<!-- installazione -->

venv e requirements.txt

# Pattern Architetturali e Principi di Design

## Architettura del Sistema

mettere nero su bianco riportando anche i design pattern utilizzati

## Code Best Practices

- **Documentazione**: Docstring complete con descrizioni di parametri, ritorni ed eccezioni
- **Error Handling**: Gestione strutturata con try/except e logging appropriato
- **Logging**: Sistema gerarchico con livelli differenziati per facilitare il troubleshooting e TRACES
- **Configurazione**: Separazione tra codice e parametri configurabili tramite variabili d'ambiente
- **Naming Convention**: Nomenclatura descrittiva in inglese standardizzata
- **Immutabilità**: Preferenza per costrutti immutabili e controllo degli effetti collaterali

## Vantaggi Abilitati dall'Architettura

**Manutenibilità**
**Estendibilità**
**Riusabilità**
**Testabilità**
**Scalabilità**
**Robustezza**
**Flessibilità**

# data preprocessing

dataset_analysis.py

NO MISSING DATA
Da survey a SUSCETTIBILITà, come da paper, sommando i delta e normalizzando (min-max scaling), min-max scaling anche per i tratti.
Rimossi i Group=Arab per incongruenza con analisi effettuate
Cercare il valore threshold per DATASET_THRESHOLD ideale

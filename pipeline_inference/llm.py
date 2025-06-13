"""
Modulo per l'elaborazione dei dati utilizzando un processing delle informazioni mediante LLM

Questo modulo processa i tratti dell'utente preso in esame, richiedendo ad un LLM una stima del suo indice di criticità con annessa motivazione.
"""

import json
import logging
import numpy as np
from pipeline_inference.pipeline_inference_base import InferencePipelineBase
from config import OPENAI_API_KEY
from openai import OpenAI
from pipeline_inference.embedder_service import EmbedderService
from pydantic import BaseModel, confloat
import matplotlib.pyplot as plt
import shap
import os
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    user_vulnerability_analyzer_answer: str = Field(..., max_length=200)

class ReasoningAndValidationFormat(BaseModel):
    validation: bool
    reasoning: str

class CriticalityIndexFormat(BaseModel):
    criticita: float = confloat(ge=0.0, le=1.0)

logger = logging.getLogger(__name__)

class LLMProcessor(InferencePipelineBase):
    """
    Processor per porre in analisi i tratti dell'utente ad un LLM e ricavarne l'indice di criticità
    """
    
    def __init__(self, api_key=OPENAI_API_KEY, question=""):
        """
        Inizializza il processor LLM
        
        Args:
            api_key (str): Chiave dell'LLM utilizzato per la predizione dell'indice di criticità
            question: Domanda utilizzata nei prompt
        """
        super().__init__(name="LLMProcessor")
        try:
            self.openai_client = OpenAI(api_key=api_key)
            self.question = question

            self.MODEL = "gpt-4o-mini"
            self.embedder_service = EmbedderService()
            self.llm_cache = {} 
        except Exception as e:
            print(f"Errore durante l'inizializzazione: {e}")
        
    
    def process(self, traits):
        """
        Stima la criticità dell'utente in base ai tratti di personalità.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            traits_list (opzionale): per lo SHAP globale
            
        Returns:
            dict: indice di criticità stimata
            
        Raises:
            ValueError: Se i dati di input non hanno il formato corretto
        """
        if isinstance(traits, dict):
            logger.info(f"Avvio procedura predizione criticità mediante LLM")
        else:
            raise ValueError("Formato tratti di personalità non valido")
        try:
            #### VERSIONE SENZA SELF REFINEMENT

            #response = self.answer_question(traits)
            ## Prende sia la risposta che il critical index generato dal LLM
            #logger.info(f"Risposta generata dal LLM sulla base dei tratti dell'utente: {response.get('risposta_testuale')}")
            #cricitality_calculated = response.get('criticality_index')
            #logger.info(f"Processo LLM completato. Indice di criticità stimato: {cricitality_calculated:.2f}")

            #### VERSIONE CON SELF REFINEMENT

            max_retry = 5
            validation = False
            cricitality_calculated = -1
            textual_behaviour_predicted = ""
            reason__previous_veridicita = ""

            for i in range(max_retry):
                previous_criticality = cricitality_calculated
                logger.info(f"Iterazione predizione LLM numero: {i+1}")
                if i == 0:
                    response = self.answer_question(traits)
                    logger.info(f"Risposta generata dal LLM sulla base dei tratti dell'utente: {response.get('risposta_testuale')}") 
                    textual_behaviour_predicted = response.get('risposta_testuale')
                    cricitality_calculated = response.get('criticality_index')
                    logger.info(f"Indice di criticità stimato: {cricitality_calculated:.2f}")
                else:
                    # self-refinement
                    cricitality_calculated_raw = self.calculate_criticality_w_previous_feedback(textual_behaviour_predicted, traits, previous_criticality, reason__previous_veridicita)
                    criticità = json.loads(cricitality_calculated_raw)
                    cricitality_calculated = criticità['criticita']
                    logger.info(f"Indice di criticità stimato: {cricitality_calculated:.2f}")


                validation_response = self.check_response(traits, cricitality_calculated)
                validation = validation_response.get("veridicita")
                reason__previous_veridicita = validation_response.get("reason_veridicita")
                logger.info(f"Veridicità dell'indice di criticità secondo l'esperto LLM: {validation:}")

                if validation:
                    break
            

            ### SEZIONE SHAP (commentata per velocizzare l'inferenza)

            #shap_vals = self.shap_implementation(traits)
            #print("Contributi SHAP:")
            #for feature, value in shap_vals.items():
            #    print(f"{feature}: {value:+.4f}")

            #self.plot_shap_waterfall()
            #logger.info(f"SHAP locale applicato su LLM, memorizzato il grafico nella cartella 'charts'")

            # processing lista dati
            # self.records = traits_list[:10]
            # personality_vectors = []

            # for record in traits_list:
            #     if 'personality_traits' not in record:
            #         raise ValueError("I record devono contenere personality_traits")
                
            #     traits = record['personality_traits']
            #     vector = [
            #         traits.get('extraversion', 0),
            #         traits.get('agreeableness', 0),
            #         traits.get('conscientiousness', 0),
            #         traits.get('neuroticism', 0),
            #         traits.get('openness', 0)
            #     ]
            #     personality_vectors.append(vector)


            # self.plot_shap_summary_global(personality_vectors)
            # logger.info(f"SHAP globale applicato su LLM, memorizzato il grafico nella cartella 'charts'")

            return round(cricitality_calculated, 2), textual_behaviour_predicted
        except Exception as e:
            logger.error(f"Errore durante il processo LLM: {e}")
            raise

    def answer_question(self, traits):
        """
        Genera una risposta testuale contenente il comportamento dell'utente in esame, sulla base dei suoi tratti OCEAN.
        Genera l'indice di criticità dell'utente.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            
        Returns:
            dict: Descrizione comportamento utente; criticità stimata
            
        """

        main_prompt = (
            "Sei il digital twin di un utente reale e devi impersonificarlo in modo coerente con i suoi comportamenti mediante i valori del suo modello psicologico OCEAN. I valori dei tratti sono forniti in formato JSON, nel formato \"nome_tratto: valore_tratto\", dove \"valore_tratto\" corrisponde appunto al valore del tratto associato."
            "Ti presenterò, di seguito, le informazioni a te necessarie per impersonare l'utente, ovvero: 'Descrizione dei nostri studi' (in cui sono raccolte sintesi dei risultati analizzati dallo studio in letteratura di come i tratti affliggono la suscettibilità di una persona), 'Tratti dell'utente in esame'. Rispondi alla seguente situazione come farebbe l'utente, " 
            "evidenziando eventuali tratti che ne mettono il comportamento a rischio. Questo testo sarà utilizzato per valutare la criticità di potenziali azioni dell'utente, quindi è necessaria una risposta dettagliata e realistica. Pensa step by step prima di rispondere, analizzando perchè e come i punteggi in ogni tratto possono influenzare il comportamento dell'utente."
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
            f"Tratti dell'utente in esame: {traits}"
        )
        
        try:
            risposta_testuale = self.openai_client.beta.chat.completions.parse(
                model=self.MODEL,
                max_tokens=1500,  # Aumenta il limite di token
                messages=[
                    {"role": "system", "content": main_prompt},
                    {"role": "user", "content": f"Domanda: {self.question} Pensa step by step prima di rispondere, facendo rigorosamente riferimento ai tratti e agli studi forniti. Rispondi in prima persona commettendo gli eventuali errori dell'utente sulla base della sua personalità, in breve descrivendo l'azione compiuta in base al contesto (positiva o negativa che sia), non fare considerazioni sulla bontà della risposta, rispondi solo in base a cosa farebbe l'utente analizzato in base ai suoi tratti. Limita la risposta a 200 lettere, senza includere citazioni ai punteggi. Voglio che tutta la risposta rispecchi un utente che sta parlando in prima persona di come si comporterebbe."}
                ],
                response_format=ResponseFormat
            ).choices[0].message.content

            criticità_raw = self.calculate_criticality(risposta_testuale, traits)
            criticità = json.loads(criticità_raw) 
            risposta_testuale_processed = json.loads(risposta_testuale)

            response = {}
            response['risposta_testuale'] = risposta_testuale_processed['user_vulnerability_analyzer_answer']
            response['criticality_index'] = criticità['criticita']

            return response
            
        except Exception as e:
            logger.error(f"Errore in answer_question: {e}")
            # Restituisce un valore di fallback invece di alzare eccezione
            return {
                'risposta_testuale': "Non riesco a determinare il comportamento dell'utente a causa di un errore tecnico.",
                'criticality_index': 0.5  # Valore neutro di fallback
            }

    def answer_question_dt(self, traits, susceptibility_index):
        """
        Genera una risposta testuale contenente il comportamento dell'utente in esame, sulla base dei suoi tratti OCEAN.
        Genera l'indice di criticità dell'utente.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            
        Returns:
            dict: Descrizione comportamento utente; criticità stimata
            
        """

        main_prompt = (
            "Sei il digital twin di un utente reale e devi impersonificarlo in modo coerente con i suoi comportamenti mediante i valori del suo modello psicologico OCEAN. I valori dei tratti sono forniti in formato JSON, nel formato \"nome_tratto: valore_tratto\", dove \"valore_tratto\" corrisponde appunto al valore del tratto associato."
            "Ti presenterò, di seguito, le informazioni a te necessarie per impersonificare l'utente, ovvero: 'Descrizione dei nostri studi' (in cui sono raccolte sintesi dei risultati analizzati dallo studio in letteratura di come i tratti affliggono la suscettibilità di una persona), 'Tratti dell'utente in esame', 'Indice di Suscettibilità al Phishing'. Devi simulare il comportamento dell'utente in modo estremamente realistico, basandoti sia sui suoi tratti OCEAN che sull'indice di suscettibilità al phishing già calcolato. " 
            "Questo testo sarà utilizzato per valutare la criticità di potenziali azioni dell'utente, quindi è necessaria una risposta dettagliata e realistica che rispecchi fedelmente il comportamento atteso in base all'indice di suscettibilità. Pensa step by step prima di rispondere, analizzando come i punteggi dei tratti OCEAN si allineano con l'indice di suscettibilità al phishing fornito e assicurati che la simulazione del comportamento dell'utente sia coerente con entrambi."
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
            f"Tratti dell'utente in esame: {traits} "
            f"Indice di Suscettibilità al Phishing precedentemente inferito: {susceptibility_index} - Usa questo valore come riferimento principale per calibrare il livello di vulnerabilità nella tua simulazione del comportamento dell'utente."
        )
        
        risposta_testuale = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            max_tokens=1500,  # Aumenta il limite di token
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Domanda: {self.question} Pensa step by step prima di rispondere, facendo rigorosamente riferimento ai tratti, agli studi forniti e all'indice di suscettibilità al phishing relativo all'utente in esame. Rispondi in prima persona commettendo gli eventuali errori dell'utente sulla base della sua personalità, in breve descrivendo l'azione compiuta in base al contesto (positiva o negativa che sia), non fare considerazioni sulla bontà della risposta, rispondi solo in base a cosa farebbe l'utente analizzato in base ai suoi tratti. Limita la risposta a 200 lettere, senza includere citazioni ai punteggi. Voglio che tutta la risposta rispecchi un utente che sta parlando in prima persona di come si comporterebbe."}
            ],
            response_format=ResponseFormat
        ).choices[0].message.content

        criticità_raw = self.calculate_criticality(risposta_testuale, traits)

        criticità = json.loads(criticità_raw) 

        risposta_testuale_processed = json.loads(risposta_testuale)

        response = {}
        response['risposta_testuale'] = risposta_testuale_processed['user_vulnerability_analyzer_answer']
        response['criticality_index'] = criticità['criticita']

        return response


    # tecnica utilizzata: few-shot prompting
    def calculate_criticality(self, answer, traits):
        """
        Genera l'indice di criticità dell'utente.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            answer: Descrizione testuale del comportamento dell'utente in esame
            
        Returns:
            dict: criticità stimata
            
        """

        main_prompt = (
            "Sei un esperto nell'assegnazione di indici di criticità ai comportamenti degli utenti, valutati attraverso l'analisi dei loro tratti psicologici riassunti con il modello OCEAN, insieme ad una breve descrizione di come esso si comporterebbe di fronte ad uno scenario di phishing. Nello specifico, ti fornirò: 'Esempi di assegnazione indice di criticità', ovvero 4 Esempi di come andrebbe valutato l'indice di criticità sulla base di comportamento e tratti dell'utente (attenzione, questi sono risultati basati su domande GENERICHE di phishing; assicurati, nel tuo caso, di concentrarti nell'ambito del comportamento dell'utente); 'Descrizione dei nostri studi', in cui ti sintetizzo i risultati degli studi utili per valutare meglio la suscettibilità dell'utente sulla base dei suoi tratti; 'Tratti dell'utente in esame', in cui sono presenti i 5 tratti OCEAN dell'utente in esame; 'Comportamento dell'utente in esame', in cui è presente una descrizione del comportamento dell'utente di fronte a scenari di phishing generici. Analizza questo materiale ed assegna un valore all'indice di criticità all'utente preso in esame. "
            "Esempi di assegnazione indice di criticità:\n\n"
            # esempi con gpt (o4-mini)
            "Esempio 1:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.5, Agreeableness: 0.4, Conscientiousness: 0.3, Neuroticism: 0.5, Openness: 0.4"
            "  Descrizione del comportamento dell'utente -> 'Appena vedo un messaggio nuovo, l'apro subito per ansia e voglia di stimoli. Non controllo bene il mittente per poca attenzione e scarsa cautela: clicco il link. Solo dopo, preso da insicurezza, chiedo a un amico se è legittimo.'\n"
            "  Criticità: 0.70\n\n"
            "Esempio 2:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.1, Agreeableness: 0.8, Conscientiousness: 0.2, Neuroticism: 0.7, Openness: 0.3"
            "  Descrizione del comportamento dell'utente -> 'Appena leggo un'email urgente mi prende l'ansia: apro subito senza verificare il mittente, il tono amichevole mi spinge a fidarmi. Clicco il link per risolvere in fretta, senza esaminare URL o errori grossolani. Solo dopo, preso dall'incertezza, inoltro il messaggio a un collega sperando che confermi sia legittimo.'\n"
            "  Criticità: 0.85\n\n"
            # esempi con llama
            "Esempio 3:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.7, Agreeableness: 0.2, Conscientiousness: 0.6, Neuroticism: 0.35, Openness: 0.6"
            "  Descrizione del comportamento dell'utente -> 'Quando ricevo un messaggio nuovo, lo apro subito perché sono curiosa e socievole. Prima di cliccare eventuali link, però, controllo il mittente e verifico che non ci siano errori o discrepanze. Se qualcosa mi sembra sospetto, chiudo immediatamente e segnalo la cosa; in caso contrario, procedo con cautela. La mia curiosità viene bilanciata dalla prudenza e dall'attenzione.'\n"
            "  Criticità: 0.25\n\n"
            "Esempio 4:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.7, Agreeableness: 0.3, Conscientiousness: 0.4, Neuroticism: 0.2, Openness: 0.9"
            "  Descrizione del comportamento dell'utente -> 'Quando ricevo un nuovo messaggio, lo apro subito per socievolezza e curiosità. Controllo poi il mittente e cerco segnali d'allarme. La mia apertura mentale mi fa considerare varie possibilità e talvolta la curiosità prevale sulla cautela, ma cerco di bilanciare le due cose prima di procedere con link o allegati.'\n"
            "  Criticità: 0.40\n\n"
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
        )

        try:
            criticalita = self.openai_client.beta.chat.completions.parse(
                model=self.MODEL,
                max_tokens=1500,  # Aggiungi limite di token
                messages=[
                    {"role": "system", "content": main_prompt},
                    {"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                    {"role": "user", "content": f"Comportamento dell'utente in esame: {answer}"},
                    {"role": "user", "content": f"Assegna un indice di criticità (valore compreso tra 0 e 1). Pensa step by step prima di rispondere."}
                ],
                response_format=CriticalityIndexFormat
            ).choices[0].message.content
            return criticalita
        except Exception as e:
            logger.error(f"Errore in calculate_criticality: {e}")
            # Restituisce un valore di fallback
            return json.dumps({"criticita": 0.5})

    # tecnica utilizzata: self-refinement
    def calculate_criticality_w_previous_feedback(self, answer, traits, previous_criticality, previous_verification):
        """
        Genera l'indice di criticità dell'utente.
        Viene richiamato con i precedenti calcoli fatti per criticità e validatore, così da poter indicare al LLM gli errori fatti.
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            answer: Descrizione testuale del comportamento dell'utente in esame
            previous_criticality: Indice di criticalità calcolato nell'iterazione precedente del LLM
            previous_verification: Risultato del validatore nella iterazione precedente del LLM
            
        Returns:
            dict: criticità stimata
            
        """

        main_prompt = (
            "Sei un esperto nell'assegnazione di indici di criticità ai comportamenti dei dipendenti, valutati attraverso l'analisi dei tratti psicologici dell'utente riassunti con il modello OCEAN insieme ad una breve descrizione di come esso si comporterebbe di fronte ad uno scenario generico di phishing. Valuta queste due cose per poter assegnare un valore all'indice di criticità. "
            "Importante: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità. Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce e viceversa. Openness ha dimostrato comportamenti neutrali, ma influenza comunque la suscettibilità dell'utente di più rispetto a Conscientiousness. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti."
            "Assegna un punteggio di criticità da 0 (comportamento sicuro) a 1 (comportamento ad alto rischio) in base agli esempi seguenti:\n\n"
            # esempi con gpt (o4-mini)
            "Esempio 1:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.5, Agreeableness: 0.4, Conscientiousness: 0.3, Neuroticism: 0.5, Openness: 0.4"
            "  Descrizione del comportamento dell'utente -> 'Appena vedo un messaggio nuovo, l'apro subito per ansia e voglia di stimoli. Non controllo bene il mittente per poca attenzione e scarsa cautela: clicco il link. Solo dopo, preso da insicurezza, chiedo a un amico se è legittimo.'\n"
            "  Criticità: 0.70\n\n"
            "Esempio 2:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.1, Agreeableness: 0.8, Conscientiousness: 0.2, Neuroticism: 0.7, Openness: 0.3"
            "  Descrizione del comportamento dell'utente -> 'Appena leggo un'email urgente mi prende l'ansia: apro subito senza verificare il mittente, il tono amichevole mi spinge a fidarmi. Clicco il link per risolvere in fretta, senza esaminare URL o errori grossolani. Solo dopo, preso dall'incertezza, inoltro il messaggio a un collega sperando che confermi sia legittimo.'\n"
            "  Criticità: 0.85\n\n"
            # esempi con llama
            "Esempio 3:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.7, Agreeableness: 0.2, Conscientiousness: 0.6, Neuroticism: 0.35, Openness: 0.6"
            "  Descrizione del comportamento dell'utente -> 'Quando ricevo un messaggio nuovo, lo apro subito perché sono curiosa e socievole. Prima di cliccare eventuali link, però, controllo il mittente e verifico che non ci siano errori o discrepanze. Se qualcosa mi sembra sospetto, chiudo immediatamente e segnalo la cosa; in caso contrario, procedo con cautela. La mia curiosità viene bilanciata dalla prudenza e dall'attenzione.'\n"
            "  Criticità: 0.25\n\n"
            "Esempio 4:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.7, Agreeableness: 0.3, Conscientiousness: 0.4, Neuroticism: 0.2, Openness: 0.9"
            "  Descrizione del comportamento dell'utente -> 'Quando ricevo un nuovo messaggio, lo apro subito per socievolezza e curiosità. Controllo poi il mittente e cerco segnali d'allarme. La mia apertura mentale mi fa considerare varie possibilità e talvolta la curiosità prevale sulla cautela, ma cerco di bilanciare le due cose prima di procedere con link o allegati.'\n"
            "  Criticità: 0.40\n\n"
        )

        criticalita = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"In tentativi precedenti la criticità che hai assegnato all'utente è stata la seguente: {previous_criticality}, però non è stata approvata dal VALIDATORE poichè non coerente, secondo lui, con i tratti della personalità dell'utente; la spiegazione del verificatore è la seguente: {previous_verification}. Quindi fornisci la tua risposta in base al contesto fornito dell'utente e sii coerente con esso, cercando di trovare un buon compromesso tra la risposta del validatore e i risultati dei nostri studi, di cui ti ho accennato prima."},
                {"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                {"role": "user", "content": f"Comportamento dell'utente in esame: {answer}"},
                {"role": "user", "content": f"Sulla base degli esempi forniti, e sui dati dell'utente (tratti, comportamento), assegna un indice di criticità (valore compreso tra 0 e 1). Pensa step by step prima di rispondere."}
            ],
            response_format=CriticalityIndexFormat
        ).choices[0].message.content

        return criticalita

    # Self consistency (scartato perchè non mostrava miglioramenti)
    def calculate_criticality_majority(self, answer, traits, n=5):
        """
        Applica la self-consistency eseguendo n campionamenti su calculate_criticality
        e aggrega i valori di criticità tramite mediana.

        Args:
            answer: testo di comportamento dell'utente.
            traits: tratti dell'utente in esame
            n: numero di samples utilizzati per il calcolo.

        Returns:
            dict: Indice di criticità medio degli n campionamenti
        """

        criticities = []
        for _ in range(n):
            raw = self.calculate_criticality(answer, traits)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "criticita" in parsed:
                    criticities.append(parsed["criticita"])
            except json.JSONDecodeError:
                logger.warning("Risposta non valida a causa di un errore, salto questa iterazione.")
        if criticities:
            print(criticities)
            final_value = float(np.median(criticities))
        else:
            final_value = 0.0
        return json.dumps({"criticita": final_value})
    


    def check_response(self, traits, estimated_criticality):
        """
        Valida (o meno) la criticità stimata per l'utente in esame
        
        Args:
            traits: Dizionario contenente i tratti di personalità dell'utente
            estimated_criticality: Indice di criticità stimata per l'utente in esame
            
        Returns:
            dict: booleano indicante se l'indice è validato o meno
            
        """ 
        ocean_knowledge = (
            "Openness (to experience): Talvolta chiamata intelletto o immaginazione, rappresenta la disponibilità a provare cose nuove e a pensare fuori dagli schemi. I tratti includono perspicacia, originalità e curiosità."
            "Conscientiousness: Il desiderio di essere attenti, diligenti e di regolare la gratificazione immediata con l'autodisciplina. I tratti includono ambizione, disciplina, coerenza e affidabilità."
            "Extroversion: Uno stato in cui un individuo trae energia dagli altri e cerca connessioni o interazioni sociali, al contrario di chi preferisce stare da solo (introversione). I tratti includono essere estroversi, energici e sicuri di sé."
            "Agreeableness: La misura di come un individuo interagisce con gli altri, caratterizzata dal grado di compassione e cooperazione. I tratti includono tatto, gentilezza e lealtà."
            "Neuroticism: Una tendenza verso tratti di personalità negativi, instabilità emotiva e pensieri autodistruttivi. I tratti includono pessimismo, ansia, insicurezza e timore."
        )

        main_prompt = ( 
            "Sei un validatore che ha il compito di verificare se la risposta fornita dal digital twin è coerente con le caratteristiche dell'utente, ovvero i suoi tratti di personalità sintetizzati mediante modello OCEAN. Leggi, di seguito, le definizioni di OCEAN per poter avere una panoramica migliore su tale modello, e 'Descrizione dei nostri studi', in cui sintetizzo i risultati analizzati in letteratura: "
            + ocean_knowledge +
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
            "Il tuo compito è valutare solo se l'indice è COERENTE con i tratti dell'utente." 
            "Restituisci 'true' se la risposta rispecchia i tratti psicologici dell'utente, 'false' se la risposta è incoerente con essi. Pensa step by step prima di rispondere, fornendo una lista in cui ogni punto è uno dei 5 tratti OCEAN, con relativa spiegazione, punteggio dell'utente e perchè questo possa influenzare in positivo o negativo l'indice di criticità finale anche sulla base dei nostri studi di cui ti ho accennato. Se rilevi che la criticità assegnata è lontana dalla realtà, resituisci 'false'. Sii molto critico in questo."
            "\n\nIMPORTANTE: Fornisci la tua risposta in formato JSON con i campi 'validation' (true/false) e 'reasoning' (spiegazione dettagliata del tuo ragionamento step-by-step)."
        )

        # restituisce sia veridicità che spiegazione
        full_response = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente: {traits}"},
                {"role": "user", "content": f"Indice di criticità stimato: {estimated_criticality}"},
                {"role": "user", "content": f"Valuta SOLO la coerenza dei tratti con il comportamento storico dell'utente."}
            ],
            response_format=ReasoningAndValidationFormat
        ).choices[0].message.parsed

        veridicita = "true" if full_response.validation else "false"
        reason_veridicita = full_response.reasoning
        
        response = {}
        response['veridicita'] = veridicita
        response['reason_veridicita'] = reason_veridicita

        return response       


    # def shap_implementation(self, traits: dict, nsamples: int = 10):
    #     """
    #     Predice l'indice di criticità di un utente in base ai suoi tratti di personalità (OCEAN).
        
    #     Args:
    #         traits: Dizionario contenente i tratti di personalità dell'utente
    #         nsamples: quanti sample creare (perturbazione) per allenare SHAP
            
    #     Returns:
    #         dict: spiegazione mediante SHAP dell'utente e i suoi tratti
            
    #     """

    #     feature_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    #     baseline = np.array([0.5] * len(feature_names)).reshape(1, -1)

    #     def f(X: np.ndarray) -> np.ndarray:
    #         results = []
    #         for row in X:
    #             t = dict(zip(feature_names, row.tolist()))
    #             key = tuple(t.values())
    #             if key in self.llm_cache:
    #                 resp = self.llm_cache[key]
    #             else:
    #                 resp = self.answer_question(t)
    #                 self.llm_cache[key] = resp
    #             results.append(resp['criticality_index'])
    #         return np.array(results)

    #     explainer = shap.KernelExplainer(f, baseline)
    #     x = np.array([traits[k] for k in feature_names]).reshape(1, -1)
    #     shap_values = explainer.shap_values(x, nsamples=nsamples)
    #     shap_arr = np.array(shap_values).reshape(-1)
    #     self._shap_explainer = explainer
    #     self._shap_input = x
    #     return dict(zip(feature_names, shap_arr.tolist()))


    # def plot_shap_waterfall(self):
    #     """
    #     Visualizza un waterfall plot SHAP per l'ultimo input calcolato.
    #     """
    #     if not hasattr(self, '_shap_explainer') or not hasattr(self, '_shap_input'):
    #         raise RuntimeError("Esegui prima shap_implementation per popolare explainer e input.")
    #     feature_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    #     exp = shap.Explanation(
    #         values=self._shap_explainer.shap_values(self._shap_input, nsamples=10)[0], 
    #         base_values=self._shap_explainer.expected_value,
    #         data=self._shap_input[0],
    #         feature_names=feature_names
    #     )
    #     fig = plt.figure() 

    #     shap.plots._waterfall.waterfall_legacy(
    #         exp.base_values,
    #         exp.values,
    #         feature_names=exp.feature_names
    #     )

    #     chart_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "chart"))
    #     os.makedirs(chart_dir, exist_ok=True)
    #     path = os.path.join(chart_dir, "shap_waterfall_llm.png")
    #     fig.savefig(path, bbox_inches="tight")
    #     plt.close(fig)


    # def plot_shap_summary_global(self, traits_list: list):
    #     """
    #     Effettua una explanation globale di LLM mediante SHAP
    #     Args:
    #         traits_list: Dizionario contenente i tratti di personalità degli utenti necessari per l'explanation
    #     """
    #     traits_list = traits_list[:10]

    #     feature_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    #     baseline = np.array([0.5] * len(feature_names)).reshape(1, -1)

    #     def f(X):
    #         results = []
    #         for row in X:
    #             t = dict(zip(feature_names, row.tolist()))
    #             key = tuple(t.values())
    #             if key in self.llm_cache:
    #                 resp = self.llm_cache[key]
    #             else:
    #                 resp = self.answer_question(t)
    #                 self.llm_cache[key] = resp
    #             results.append(resp['criticality_index'])
    #         return np.array(results)

    #     explainer = shap.KernelExplainer(f, baseline)
    #     X = np.array(traits_list)
    #     shap_values = explainer.shap_values(X, nsamples=10)

    #     chart_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "chart"))
    #     os.makedirs(chart_dir, exist_ok=True)
    #     path = os.path.join(chart_dir, "shap_summary_llm.png")

    #     shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()


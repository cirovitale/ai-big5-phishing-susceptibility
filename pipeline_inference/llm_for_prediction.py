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
from embedder_service import EmbedderService
from pydantic import BaseModel, confloat
import matplotlib.pyplot as plt
import shap

class ResponseFormat(BaseModel):
    user_vulnerability_analyzer_answer: str

class ValidationFormat(BaseModel):
    validation: bool

class CriticalityIndexFormat(BaseModel):
    criticita: float = confloat(ge=0.0, le=1.0)

logger = logging.getLogger(__name__)

class LLMPrediction(InferencePipelineBase):
    """
    Processor per porre in analisi i tratti dell'utente ad un LLM e ricavarne l'indice di criticità
    """
    
    def __init__(self, api_key=OPENAI_API_KEY):
        """
        Inizializza il processor LLM
        
        Args:
            api_key (str): Chiave dell'LLM utilizzato per la predizione dell'indice di criticità
        """
        super().__init__(name="LLMPrediction")
        try:
            self.openai_client = OpenAI(api_key=api_key)

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
                #response_processed = json.loads(validation_response) 
                #validation = response_processed["validation"]
                validation = validation_response.get("veridicita")
                reason__previous_veridicita = validation_response.get("reason_veridicita")
                logger.info(f"Veridicità dell'indice di criticità secondo l'esperto LLM: {validation:}")

                if validation:
                    break
            

            # serve per stimare quanto ogni tratto influenza il calcolo finale
            shap_vals = self.shap_implementation(traits)
            print("Contributi SHAP:")
            for feature, value in shap_vals.items():
                print(f"{feature}: {value:+.4f}")

            # Visualizza grafico (serve matplotlib e SHAP installati)
            self.plot_shap_waterfall()


            return round(cricitality_calculated, 2)
        except Exception as e:
            logger.error(f"Errore durante il processo LLM: {e}")
            raise

    def answer_question(self, traits):

        main_prompt = (
            "Sei il digital twin di un utente reale e devi impersonificarlo in modo coerente con i suoi comportamenti mediante i valori del suo modello psicologico OCEAN. I valori dei tratti sono forniti in formato JSON, nel formato \"nome_tratto: valore_tratto\", dove \"valore_tratto\" corrisponde appunto al valore del tratto associato."
            "Ti presenterò, di seguito, le informazioni a te necessarie per impersonare l'utente, ovvero: 'Descrizione dei nostri studi' (in cui sono raccolte sintesi dei risultati analizzati dallo studio in letteratura di come i tratti affliggono la suscettibilità di una persona), 'Tratti dell'utente in esame'. Rispondi alla seguente situazione come farebbe l'utente, " 
            "evidenziando eventuali tratti che ne mettono il comportamento a rischio. Questo testo sarà utilizzato per valutare la criticità di potenziali azioni dell'utente, quindi è necessaria una risposta dettagliata e realistica. Pensa step by step prima di rispondere, analizzando perchè e come i punteggi in ogni tratto possono influenzare il comportamento dell'utente."
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
            f"Tratti dell'utente in esame: {traits}"
        )
        
        risposta_testuale = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                #{"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                {"role": "user", "content": f"Domanda: Come ti poni rispetto ad una nuova mail nella tua casella di posta, o un nuovo messaggio sul tuo servizio di messaggistica preferito? Pensa step by step prima di rispondere, facendo rigorosamente riferimento ai tratti e agli studi forniti. Rispondi in prima persona commettendo gli eventuali errori dell'utente sulla base della sua personalità, in breve descrivendo l'azione compiuta in base al contesto (positiva o negativa che sia), non fare considerazioni sulla bontà della risposta, rispondi solo in base a cosa farebbe l'utente analizzato in base ai suoi tratti. Limita la risposta a 400 lettere, senza includere citazioni ai punteggi. Voglio che tutta la risposta rispecchi un utente che sta parlando in prima persona di come si comporterebbe."}
            ],
            response_format=ResponseFormat
        ).choices[0].message.content

        criticità_raw = self.calculate_criticality(risposta_testuale, traits)
        #criticità_raw = self.calculate_criticality_no_knowledge_studies(risposta_testuale, traits)
        
        #criticità_raw = self.calculate_criticality_majority(risposta_testuale, traits)
        criticità = json.loads(criticità_raw) 


        response = {}
        response['risposta_testuale'] = risposta_testuale
        response['criticality_index'] = criticità['criticita']

        return response


    # tecnica utilizzata: few-shot prompting
    def calculate_criticality(self, answer, traits):


        main_prompt = (
            "Sei un esperto nell'assegnazione di indici di criticità ai comportamenti degli utenti, valutati attraverso l'analisi dei loro tratti psicologici riassunti con il modello OCEAN, insieme ad una breve descrizione di come esso si comporterebbe di fronte ad uno scenario generico di phishing. Nello specifico, ti fornirò: 'Esempi di assegnazione indice di criticità', ovvero 4 Esempi di come andrebbe valutato l'indice di criticità sulla base di comportamento e tratti dell'utente; 'Descrizione dei nostri studi', in cui ti sintetizzo i risultati degli studi utili per valutare meglio la suscettibilità dell'utente sulla base dei suoi tratti; 'Tratti dell'utente in esame', in cui sono presenti i 5 tratti OCEAN dell'utente in esame; 'Comportamento dell'utente in esame', in cui è presente una descrizione del comportamento dell'utente di fronte a scenari di phishing generici. Analizza questo materiale ed assegna un valore all'indice di criticità all'utente preso in esame. "
            "Esempi di assegnazione indice di criticità:\n\n"
            # esempi con gpt (o4-mini)
            "Esempio 1:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.5, Agreeableness: 0.4, Conscientiousness: 0.3, Neuroticism: 0.5, Openness: 0.4"
            "  Descrizione del comportamento dell'utente -> 'Appena vedo un messaggio nuovo, l’apro subito per ansia e voglia di stimoli. Non controllo bene il mittente per poca attenzione e scarsa cautela: clicco il link. Solo dopo, preso da insicurezza, chiedo a un amico se è legittimo.'\n"
            "  Criticità: 0.70\n\n"
            "Esempio 2:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.1, Agreeableness: 0.8, Conscientiousness: 0.2, Neuroticism: 0.7, Openness: 0.3"
            "  Descrizione del comportamento dell'utente -> 'Appena leggo un’email urgente mi prende l’ansia: apro subito senza verificare il mittente, il tono amichevole mi spinge a fidarmi. Clicco il link per risolvere in fretta, senza esaminare URL o errori grossolani. Solo dopo, preso dall’incertezza, inoltro il messaggio a un collega sperando che confermi sia legittimo.'\n"
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

        criticalita = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                {"role": "user", "content": f"Comportamento dell'utente in esame: {answer}"},
                {"role": "user", "content": f"Assegna un indice di criticità (valore compreso tra 0 e 1). Pensa step by step prima di rispondere."}
            ],
            response_format=CriticalityIndexFormat
        ).choices[0].message.content

        return criticalita

    # tecnica utilizzata: self-refinement
    def calculate_criticality_w_previous_feedback(self, answer, traits, previous_criticality, previous_verification):


        main_prompt = (
            "Sei un esperto nell'assegnazione di indici di criticità ai comportamenti dei dipendenti, valutati attraverso l'analisi dei tratti psicologici dell'utente riassunti con il modello OCEAN insieme ad una breve descrizione di come esso si comporterebbe di fronte ad uno scenario generico di phishing. Valuta queste due cose per poter assegnare un valore all'indice di criticità. "
            "Importante: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità. Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce e viceversa. Openness ha dimostrato comportamenti neutrali, ma influenza comunque la suscettibilità dell'utente di più rispetto a Conscientiousness. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti."
            "Assegna un punteggio di criticità da 0 (comportamento sicuro) a 1 (comportamento ad alto rischio) in base agli esempi seguenti:\n\n"
            # esempi con gpt (o4-mini)
            "Esempio 1:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.5, Agreeableness: 0.4, Conscientiousness: 0.3, Neuroticism: 0.5, Openness: 0.4"
            "  Descrizione del comportamento dell'utente -> 'Appena vedo un messaggio nuovo, l’apro subito per ansia e voglia di stimoli. Non controllo bene il mittente per poca attenzione e scarsa cautela: clicco il link. Solo dopo, preso da insicurezza, chiedo a un amico se è legittimo.'\n"
            "  Criticità: 0.70\n\n"
            "Esempio 2:\n"
            "  Tratti OCEAN dell'utente -> Extraversion: 0.1, Agreeableness: 0.8, Conscientiousness: 0.2, Neuroticism: 0.7, Openness: 0.3"
            "  Descrizione del comportamento dell'utente -> 'Appena leggo un’email urgente mi prende l’ansia: apro subito senza verificare il mittente, il tono amichevole mi spinge a fidarmi. Clicco il link per risolvere in fretta, senza esaminare URL o errori grossolani. Solo dopo, preso dall’incertezza, inoltro il messaggio a un collega sperando che confermi sia legittimo.'\n"
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

    # tecnica utilizzata: zero-shot
    def calculate_criticality_no_knowledge_studies(self, answer, traits):

        main_prompt = (
            "Sei un esperto nell'assegnazione di indici di criticità ai comportamenti dei dipendenti, valutati attraverso l'analisi dei tratti psicologici dell'utente riassunti con il modello OCEAN insieme ad una breve descrizione di come esso si comporterebbe di fronte ad uno scenario generico di phishing. Valuta queste due cose per poter assegnare un valore all'indice di criticità. "
        )

        criticalita = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                {"role": "user", "content": f"Comportamento dell'utente in esame: {answer}"},
                {"role": "user", "content": f"Sulla base degli esempi forniti, e sui dati dell'utente (tratti, comportamento), assegna un indice di criticità (valore compreso tra 0 e 1). Pensa step by step prima di rispondere."}
            ],
            response_format=CriticalityIndexFormat
        ).choices[0].message.content

        print(self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente in esame: {traits}"},
                {"role": "user", "content": f"Comportamento dell'utente in esame: {answer}"},
                {"role": "user", "content": f"Sulla base degli esempi forniti, e sui dati dell'utente (tratti, comportamento), assegna un indice di criticità (valore compreso tra 0 e 1). Pensa step by step prima di rispondere."}
            ],
        ).choices[0].message.content)

        return criticalita

    # Self consistency
    def calculate_criticality_majority(self, answer, traits, n=5):
        """
        Applica la self-consistency eseguendo n campionamenti su calculate_criticality
        e aggrega i valori di criticità tramite mediana.

        Args:
            answer (str): testo di comportamento dell'utente.
            n (int): numero di traiettorie di sampling.

        Ritorna:
            str: JSON con {"criticita": valore_mediano}.
        """
        criticities = []
        for _ in range(n):
            raw = self.calculate_criticality(answer, traits)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "criticita" in parsed:
                    criticities.append(parsed["criticita"])
            except json.JSONDecodeError:
                logger.warning("Risposta critica non valida, salto questa iterazione.")
        if criticities:
            print(criticities)
            final_value = float(np.median(criticities))
        else:
            final_value = 0.0
        return json.dumps({"criticita": final_value})
    


    def check_response(self, traits, estimated_criticality):

        ocean_knowledge = (
            "Openness (to experience): Talvolta chiamata intelletto o immaginazione, rappresenta la disponibilità a provare cose nuove e a pensare fuori dagli schemi. I tratti includono perspicacia, originalità e curiosità."
            "Conscientiousness: Il desiderio di essere attenti, diligenti e di regolare la gratificazione immediata con l’autodisciplina. I tratti includono ambizione, disciplina, coerenza e affidabilità."
            "Extroversion: Uno stato in cui un individuo trae energia dagli altri e cerca connessioni o interazioni sociali, al contrario di chi preferisce stare da solo (introversione). I tratti includono essere estroversi, energici e sicuri di sé."
            "Agreeableness: La misura di come un individuo interagisce con gli altri, caratterizzata dal grado di compassione e cooperazione. I tratti includono tatto, gentilezza e lealtà."
            "Neuroticism: Una tendenza verso tratti di personalità negativi, instabilità emotiva e pensieri autodistruttivi. I tratti includono pessimismo, ansia, insicurezza e timore."
        )

        main_prompt = ( ##TODO capire se questo punto va bene così o si deve dare una motivazione più dettagliata per constatare la veridicità dell'indice
            "Sei un validatore che ha il compito di verificare se la risposta fornita dal digital twin è coerente con le caratteristiche dell'utente, ovvero i suoi tratti di personalità sintetizzati mediante modello OCEAN. Leggi, di seguito, le definizioni di OCEAN per poter avere una panoramica migliore su tale modello, e 'Descrizione dei nostri studi', in cui sintetizzo i risultati analizzati in letteratura: "
            + ocean_knowledge +
            "Descrizione dei nostri studi: dai nostri studi, è risultato che un aumento nei valori dei tratti Agreeableness, Neuroticism, Extraversion (in questo ordine) comportano anche una crescita della suscettibilità al phishing dell'utente (di conseguenza, una loro diminuzione comporta una minore suscettibilità). Conscientiousness invece dimostra il comportamento opposto, ovvero che al suo crescere la suscettibilità dell'utente si riduce. Openness ha dimostrato comportamenti neutrali, leggermente tendenti verso il far crescere la suscettibilità dell'utente insieme alla loro crescita, anche se in maniera minore degli altri 3 tratti citati. Ricorda queste informazioni quando andrai ad analizzare la situazione dell'utente e applica tali nozioni sui suoi tratti in modo non banale."
            "Il tuo compito è valutare solo se l'indice è COERENTE con i tratti dell'utente." 
            "Restituisci 'true' se la risposta rispecchia i tratti psicologici dell'utente, 'false' se la risposta è incoerente con essi. Pensa step by step prima di rispondere, fornendo una lista in cui ogni punto è uno dei 5 tratti OCEAN, con relativa spiegazione, punteggio dell'utente e perchè questo possa influenzare in positivo o negativo l'indice di criticità finale anche sulla base dei nostri studi di cui ti ho accennato. Se rilevi che la criticità assegnata è lontana dalla realtà, resituisci 'false'. Sii molto critico in questo."
        )

        veridicità = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente: {traits}"},
                {"role": "user", "content": f"Indice di criticità stimato: {estimated_criticality}"},
                {"role": "user", "content": f"Valuta SOLO la coerenza dei tratti con il comportamento storico dell'utente."}
            ],
            response_format=ValidationFormat
        ).choices[0].message.content

        # per capire se e perchè dà falso alla veridicità
        reason_veridicita = self.openai_client.beta.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": main_prompt},
                {"role": "user", "content": f"Tratti dell'utente: {traits}"},
                {"role": "user", "content": f"Indice di criticità stimato: {estimated_criticality}"},
                {"role": "user", "content": f"Hai valutato questa risposta come {veridicità}. Come mai? Spiega il tuo ragionamento."}
            ],
        ).choices[0].message.content

        response = {}
        response['veridicita'] = veridicità
        response['reason_veridicita'] = reason_veridicita

        return response       



    def shap_implementation(self, traits: dict, nsamples: int = 10):
        feature_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        baseline = np.array([0.5] * len(feature_names)).reshape(1, -1)

        def f(X: np.ndarray) -> np.ndarray:
            results = []
            for row in X:
                t = dict(zip(feature_names, row.tolist()))
                key = tuple(t.values())
                if key in self.llm_cache:
                    resp = self.llm_cache[key]
                else:
                    resp = self.answer_question(t)
                    self.llm_cache[key] = resp
                results.append(resp['criticality_index'])
            return np.array(results)

        explainer = shap.KernelExplainer(f, baseline)
        x = np.array([traits[k] for k in feature_names]).reshape(1, -1)
        shap_values = explainer.shap_values(x, nsamples=nsamples)
        shap_arr = np.array(shap_values).reshape(-1)
        self._shap_explainer = explainer
        self._shap_input = x
        return dict(zip(feature_names, shap_arr.tolist()))

    def plot_shap(self, shap_values: dict):
        """
        Visualizza i valori SHAP con un bar chart.

        Args:
            shap_values (dict): mapping trait -> shap value
        """
        names = list(shap_values.keys())
        values = list(shap_values.values())
        plt.figure(figsize=(6,4))
        plt.bar(names, values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("SHAP value")
        plt.title("SHAP feature contributions")
        plt.tight_layout()
        plt.show()

    def plot_shap_waterfall(self):
        """
        Visualizza un waterfall plot SHAP per l'ultimo input calcolato.
        """
        if not hasattr(self, '_shap_explainer') or not hasattr(self, '_shap_input'):
            raise RuntimeError("Esegui prima shap_implementation per popolare explainer e input.")
        exp = shap.Explanation(
            values=self._shap_explainer.shap_values(self._shap_input, nsamples=10)[0],  # coerenza con default
            base_values=self._shap_explainer.expected_value,
            data=self._shap_input[0],
            feature_names=["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
        )
        shap.waterfall_plot(exp)
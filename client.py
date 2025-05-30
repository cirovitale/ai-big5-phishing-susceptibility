"""
Test semplificato delle API del server Flask
"""
import requests
import json
from datetime import datetime

# URL base del server
BASE_URL = "http://127.0.0.1:5000"

def predict():
    input_data = {
        "traits": {
            "extraversion": 0.1,
            "agreeableness": 0.1,
            "conscientiousness": 0.9,
            "neuroticism": 0.2,
            "openness": 0.2
        },
        "cf": "test_user_1",
        "question_for_prompt": "Come ti comporteresti in caso di messaggio su whatsapp su un gruppo non ufficiale? Ti vengono chiesti dei documenti riservati."
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def extract():
    input_data = {
        "process_form": False,
        "process_excel": True
    }
    
    response = requests.post(
        f"{BASE_URL}/extract",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def testing():
    input_data = {
        "question_for_prompt": "Come ti comporteresti in caso di messaggio su whatsapp su un gruppo non ufficiale? Ti vengono chiesti dei documenti riservati."
    }
    response = requests.post(f"{BASE_URL}/testing", json=input_data)
    print(json.dumps(response.json(), indent=2))

def training():
    response = requests.post(f"{BASE_URL}/training")
    print(json.dumps(response.json(), indent=2))

def register_digital_twin():
    input_data = {
        "cf": "RSSMRA80L05F593A",
        "first_name": "Mario",
        "last_name": "Rossi",
        "traits": {
            "extraversion": 0.75,
            "agreeableness": 0.65,
            "conscientiousness": 0.85,
            "neuroticism": 0.25,
            "openness": 0.70
        },
        "creation_datetime": datetime.now().isoformat(),
        "last_update_datetime": datetime.now().isoformat()
    }

    response = requests.post(
        f"{BASE_URL}/digital-twin/register",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def predict_by_digital_twin():
    input_data = {
        "cf": "RSSMRA80L05F593A",
        "question_for_prompt": "Come ti comporteresti in caso di messaggio su whatsapp su un gruppo non ufficiale? Ti vengono chiesti dei documenti riservati."
    }

    response = requests.post(
        f"{BASE_URL}/digital-twin/predict",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def get_digital_twin():
    cf = "RSSMRA80L05F593A"
    response = requests.get(f"{BASE_URL}/digital-twin/{cf}")
    print(json.dumps(response.json(), indent=2))


def run_all_tests():
    """Esegue tutti i test delle chiamate presenti"""

    # print("\n1. Extract")
    # extract()
    
    # print("\n2. Training")
    # training()
    
    print("\n3. Predict")
    predict()

    # print("\n4. Testing")
    # testing()

    #print("\n5. Register Digital Twin")
    #register_digital_twin()

    #print("\n6. Get Digital Twin")
    #get_digital_twin()

    #print("\n7. Predict by Digital Twin")
    #predict_by_digital_twin()

if __name__ == "__main__":
    run_all_tests()
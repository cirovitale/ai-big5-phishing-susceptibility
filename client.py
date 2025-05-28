"""
Test semplificato delle API del server Flask
"""
import requests
import json

# URL base del server
BASE_URL = "http://127.0.0.1:5000"

def test_predict():
    input_data = {
        "traits": {
            "extraversion": 0.8,
            "agreeableness": 0.7,
            "conscientiousness": 0.9,
            "neuroticism": 0.2,
            "openness": 0.6
            },
        "user_id": "test_user_1"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def test_extract():
    input_data = {
        "process_form": False,
        "process_excel": True
    }
    
    response = requests.post(
        f"{BASE_URL}/extract",
        json=input_data
    )
    print(json.dumps(response.json(), indent=2))

def test_testing():
    response = requests.get(f"{BASE_URL}/testing")
    print(json.dumps(response.json(), indent=2))

def run_all_tests():
    """Esegue tutti i test delle chiamate presenti"""
    
    print("\n1. Test Predict")
    test_predict()
    
    # print("\n2. Test Extract")
    # test_extract()
    
    #print("\n3. Test Testing")
    #test_testing()

if __name__ == "__main__":
    run_all_tests()
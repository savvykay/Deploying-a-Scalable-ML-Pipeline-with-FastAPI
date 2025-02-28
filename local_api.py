import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"

r = requests.get(url)

# print the status code and welcome message
print(f"GET Request - Status Code: {r.status_code}")
print(f"Response: {r.json()}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to the inference endpoint
post_url = "http://127.0.0.1:8000/data/"
headers = {"Content-Type": "application/json"}

r = requests.post(post_url, data=json.dumps(data), headers=headers)

# Print the status code and result of inference
print(f"\nPOST Request - Status Code: {r.status_code}")
print(f"Prediction: {r.json()}")

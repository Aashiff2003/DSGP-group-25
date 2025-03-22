import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict_alert"

# Define the input data as a dictionary
input_data = {
    "NumberStruckActual": 2,
    "WildlifeSize": 1,
    "ConditionsSky_No Cloud": 0,
    "ConditionsSky_Overcast": 1,
    "ConditionsSky_Some Cloud": 0
}

# Send the POST request
response = requests.post(url, json=input_data)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())


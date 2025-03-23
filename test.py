import requests
import os

# ✅ API URL (Ensure Flask is running on this port)
API_URL = "http://127.0.0.1:5001/predict"

# ✅ Correct image file path
IMAGE_PATH = "/Users/akshankumarsen/Downloads/drops-of-rain-on-the-window-blurred-trees-in-the-2024-12-01-11-44-37-utc.jpg"

# ✅ Check if the file exists before making the request
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image file not found: {IMAGE_PATH}")
    exit(1)

try:
    print("📤 Sending request to API...")
    with open(IMAGE_PATH, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(API_URL, files=files)

    print("📩 Response received!")

    # ✅ Print API response
    try:
        response_data = response.json()
        print("✅ API Response:", response_data)
    except requests.exceptions.JSONDecodeError:
        print("❌ Failed to decode JSON. Raw response:", response.text)

except requests.exceptions.ConnectionError:
    print(f"❌ Could not connect to the API at {API_URL}. Make sure Flask is running.")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")


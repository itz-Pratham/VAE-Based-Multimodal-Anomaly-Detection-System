import requests
import time
import random

# CHANGE THIS: Match your local uvicorn address
# Use 'predict' or 'api/predict' depending on what is in your main.py
API_URL = "http://127.0.0.1:8000/api/predict" 

def run_simulation():
    print(f"📡 Simulating traffic to: {API_URL}")
    while True:
        payload = {
            "values": [random.uniform(0.1, 0.9) for _ in range(14)],
            "timestamp": time.time()
        }
        try:
            res = requests.post(API_URL, json=payload, timeout=2)
            if res.status_code == 200:
                print(f"✅ Sent: {res.json()['score']:.4f}")
            else:
                print(f"❌ Server Error: {res.status_code}")
        except Exception as e:
            print(f"📡 Connection Failed: Check if Uvicorn is running on port 8000")
        
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()
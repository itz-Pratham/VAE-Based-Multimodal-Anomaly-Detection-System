from fastapi import FastAPI
from pydantic import BaseModel
import time
import random

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For testing only
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    values: list[float]

history_log = []

@app.post("/api/predict")
async def predict(data: InputData):
    score = random.uniform(0.1, 0.9)
    is_anomaly = score > 0.8
    record = {
        "timestamp": time.time(),
        "score": score,
        "is_anomaly": is_anomaly,
        "latency_ms": random.uniform(10, 50)
    }
    history_log.append(record)
    if len(history_log) > 50:
        history_log.pop(0)
    return record

@app.get("/api/history")
async def get_history():
    return history_log
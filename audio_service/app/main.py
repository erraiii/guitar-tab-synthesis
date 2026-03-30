from fastapi import FastAPI, UploadFile, File
import numpy as np
from inference import run_inference_py

app = FastAPI(title="Basic Pitch Inference Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    audio_bytes = await file.read()
    result = run_inference_py(audio_bytes)
    return result
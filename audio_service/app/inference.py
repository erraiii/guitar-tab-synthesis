from basic_pitch.inference import predict
import numpy as np
import tempfile

def run_inference(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model_output, midi_data, note_events = predict(tmp_path)

    return note_events


def run_inference_py(audio_bytes):
    raw = run_inference(audio_bytes)

    result = []
    for start, end, pitch, confidence, _ in raw:
        result.append({
            "start": float(start),
            "end": float(end),
            "pitch": int(pitch),
            "confidence": float(confidence)
        })

    return {"notes": result}

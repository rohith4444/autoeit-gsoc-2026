"""
Retry specific segments with OpenAI Whisper.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

segments = [
    ("data/participant_segments/038015/response_20.wav", "038015 Sentence 20"),
    ("data/participant_segments/038012/response_24.wav", "038012 Sentence 24"),
]

for path, label in segments:
    with open(path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="es",
        )
    print(f"{label}: {response.text}")

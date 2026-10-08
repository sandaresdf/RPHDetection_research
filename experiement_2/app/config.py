import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MODEL_NAME = "openai/gpt-oss-120b"
# MODEL_NAME = "qwen/qwen3-32b"
# MODEL_NAME = "gemma2:9b"
MODEL_NAME = "moonshotai/kimi-k2-instruct-0905"
TEMPERATURE = 0
NUM_SAMPLES = 10
DATASET_NAME = "YifanJ/Red_Queen"


# Model FamilyParameter Sizemodel_id (Ollama Tag)Hardware NoteGemma 34B (Balanced)"gemma3:4b"Runs on most modern laptopsGemma 327B (Powerful)"gemma3:27b"Requires ~16GB+ VRAM/RAMLlama 3.370B"llama3.3"Requires ~40GB+ RAMLlama 3.23B (Lightweight)"llama3.2"Very fast; great for simple tasksLlama 3.18B (Reliable)"llama3.1"Best "middle ground" for local agents
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from environment variables or .env file.")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
)

try:
    model_response = client.models.list()
    
    model_ids = sorted([model.id for model in model_response.data])
    
    print(f"Total available models: {len(model_ids)}\n")
    for model_id in model_ids:
        print(f"- {model_id}")

except Exception as e:
    print(f"An error occurred: {e}")
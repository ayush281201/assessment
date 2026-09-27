import json
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        env_path = Path(__file__).resolve().parent / ".env"
        if not env_path.exists():
            return False

        values = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"\'')

        os.environ.setdefault(key, value)
        for key, value in values.items():
            os.environ.setdefault(key, value)
        return True

from src.agent import create_qa

load_dotenv()

logger = logging.getLogger("QAGenerator")
logger.setLevel(logging.INFO)


def run_pipeline(topic: str, complexity: str, question_count: int):
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from environment variables or .env file.")

    if question_count <= 0:
        raise ValueError("question_count must be greater than zero.")

    logger.info(f"Initializing AutoGen Agents for Topic: '{topic}' [{complexity}]...")
    assistant, proxy = create_qa()

    prompt = (
        f"""Generate exactly {question_count} distinct questions about the topic '{topic}' at an '{complexity}' complexity level.
        For each item, return valid JSON in this exact structure:
        {{
          "questions": [
            {{
              'question': '...',
              'answer': '...',
              'reference_url': '...'
            }}
          ]
        }}
        Important rules:
        - Use technical, accurate content.
        - Include one verified reference URL per question.
        - Call the verify_reference tool for each URL before finalizing it.
        - If any URL is not valid, replace it and verify again.
        - End your final response with TERMINATE."""
    )

    logger.info("[*] Starting conversation loop and tool executions...\n")

    chat_result = proxy.initiate_chat(
        recipient=assistant,
        message=prompt,
        max_turns=6,
    )

    result = getattr(chat_result, "summary", None) or str(chat_result)
    return result


if __name__ == "__main__":
    sample_payload = {
        "topic": "AgenticAI",
        "complexity": "Advanced",
        "question_count": 2,
    }

    try:
        result = run_pipeline(**sample_payload)
        print(json.dumps(result, indent=4, ensure_ascii=False)) if isinstance(result, (dict, list)) else print(result)
    except Exception as e:
        logger.error(f"[!] Execution failed: {str(e)}")
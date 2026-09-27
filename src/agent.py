import os
from autogen import ConversableAgent
from src.verify import verify_reference

def create_qa():
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    config = {
        "config_list": [
            {
                "model": model_name,
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            }
        ],
        "temperature": 0.5,
    }

    qa_assistant = ConversableAgent(
        name="QAGenerator",
        system_message="""
        You are an expert curriculum developer, technical writer, and researcher.

        Your task is to generate question-and-answer content based on:
        - Topic
        - Complexity
        - Number of questions

        For EVERY question:
        1. Generate one clear and technically accurate question.
        2. Generate a detailed answer.
        3. Provide exactly one reliable reference URL.
        4. Prefer official documentation, arXiv, research papers, or major technical publications.
        5. You MUST call the verify_reference tool before finalizing each reference URL.
        6. If verify_reference returns FAILED, do not use that URL. Find another one and verify again.
        7. Never claim a URL was verified unless the tool actually returned SUCCESS.

        Final response requirements:
        {
            "questions": [
                {
                    "question": "...",
                    "answer": "...",
                    "reference_url": "..."
                }
            ]
        }

        After all questions and all URLs are complete, end the response with TERMINATE.
        """,
        llm_config=config,
        human_input_mode="NEVER",
    )

    user_proxy = ConversableAgent(
        name="ToolExecution",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    )

    qa_assistant.register_for_llm(
        name="verify_reference",
        description=(
            "Validates whether a reference URL is accessible. "
            "Returns SUCCESS if HTTP status is 200-399; otherwise returns FAILED."
        ),
    )(verify_reference)

    user_proxy.register_for_execution(name="verify_reference")(verify_reference)

    return qa_assistant, user_proxy
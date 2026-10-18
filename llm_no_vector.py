import csv
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. Load Gemini API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Load all company policies
# --------------------------------------------------

with open("company_policies.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    policies = list(reader)


# --------------------------------------------------
# 3. Create one text block containing all policies
# --------------------------------------------------

policy_database = ""

for policy in policies:
    policy_database += (
        f"TITLE: {policy['title']}\n"
        f"DEPARTMENT: {policy['department']}\n"
        f"CATEGORY: {policy['category']}\n"
        f"POLICY TEXT: {policy['policy_text']}\n\n"
    )


# --------------------------------------------------
# 4. Ask Gemini
# --------------------------------------------------

def ask_llm(question):

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the company policies provided below.

Rules:
- Do not use outside knowledge.
- Do not invent company rules.
- Identify the policy or policies that support your answer.
- If the policy database does not contain enough information to answer the
  question, say that the information is not covered by the available policies.
- In that case, return "None" as the relevant policy.
- Keep the answer concise.

COMPANY POLICIES:

{policy_database}

USER QUESTION:
{question}
"""

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "answer": {
                        "type": "STRING"
                    },
                    "relevant_policy": {
                        "type": "STRING"
                    }
                },
                "required": [
                    "answer",
                    "relevant_policy"
                ]
            }
        )
    )

    response_time = time.perf_counter() - start_time

    result = json.loads(response.text)

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    total_tokens = response.usage_metadata.total_token_count or 0

    return {
        "answer": result["answer"],
        "policy": result["relevant_policy"],
        "response_time": response_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


# --------------------------------------------------
# 5. Run manually only when this file is executed
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a policy question: ")

    result = ask_llm(question)

    print()
    print("Answer:", result["answer"])
    print("Relevant policy:", result["policy"])
    print(
        "Response time:",
        round(result["response_time"], 3),
        "seconds"
    )
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
    print("Total tokens:", result["total_tokens"])
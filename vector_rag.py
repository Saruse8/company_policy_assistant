import csv
import json
import os
import time

import numpy as np
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
# 2. Load company policies
# --------------------------------------------------

with open("company_policies.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    policies = list(reader)


# --------------------------------------------------
# 3. Load previously created policy embeddings
# --------------------------------------------------

policy_embeddings = np.load("policy_embeddings.npy")

if len(policy_embeddings) != len(policies):
    raise ValueError(
        "The number of saved embeddings does not match "
        "the number of policies."
    )


# --------------------------------------------------
# 4. Cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )


# --------------------------------------------------
# 5. Retrieve the most relevant policies
# --------------------------------------------------

def retrieve_policies(question, top_k=3):

    # Create an embedding for the user's question
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    question_embedding = np.array(
        response.embeddings[0].values
    )

    similarities = []

    # Compare the question with every policy embedding
    for index, policy_embedding in enumerate(policy_embeddings):

        score = cosine_similarity(
            question_embedding,
            policy_embedding
        )

        similarities.append(
            (index, score)
        )


    # Sort from highest similarity to lowest similarity
    similarities.sort(
        key=lambda item: item[1],
        reverse=True
    )


    # Keep only the top results
    top_results = similarities[:top_k]

    retrieved_policies = []

    for index, score in top_results:

        retrieved_policies.append({
            "policy": policies[index],
            "similarity": score
        })


    return retrieved_policies


# --------------------------------------------------
# 6. Ask Gemini using only retrieved policies
# --------------------------------------------------

def ask_vector_llm(question):

    start_time = time.perf_counter()


    # Retrieve the 3 most relevant policies
    retrieved = retrieve_policies(
        question,
        top_k=3
    )


    # Create context containing only retrieved policies
    context = ""

    for item in retrieved:

        policy = item["policy"]

        context += (
            f"TITLE: {policy['title']}\n"
            f"DEPARTMENT: {policy['department']}\n"
            f"CATEGORY: {policy['category']}\n"
            f"POLICY TEXT: {policy['policy_text']}\n\n"
        )


    # Prompt for Gemini
    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the company policies provided below.

Rules:
- Do not use outside knowledge.
- Do not invent company rules.
- Identify the policy or policies that support your answer.
- If the provided policies do not contain enough information to answer the
  question, say that the information is not covered by the available policies.
- In that case, return "None" as the relevant policy.
- Keep the answer concise.

RETRIEVED COMPANY POLICIES:

{context}

USER QUESTION:
{question}
"""


    # Ask Gemini
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


    # Stop measuring response time
    response_time = time.perf_counter() - start_time


    # Convert JSON response into Python data
    result = json.loads(response.text)


    # Token usage for the LLM generation call
    input_tokens = (
        response.usage_metadata.prompt_token_count or 0
    )

    output_tokens = (
        response.usage_metadata.candidates_token_count or 0
    )

    total_tokens = (
        response.usage_metadata.total_token_count or 0
    )


    return {
        "answer": result["answer"],
        "policy": result["relevant_policy"],
        "response_time": response_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "retrieved": retrieved
    }


# --------------------------------------------------
# 7. Run manually only when this file is executed
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a policy question: ")

    result = ask_vector_llm(question)


    print()

    print("Retrieved policies:")

    for item in result["retrieved"]:

        print(
            "-",
            item["policy"]["title"],
            "| Similarity:",
            round(item["similarity"], 3)
        )


    print()

    print("Answer:", result["answer"])
    print("Relevant policy:", result["policy"])

    print(
        "Response time:",
        round(result["response_time"], 3),
        "seconds"
    )

    print(
        "Input tokens:",
        result["input_tokens"]
    )

    print(
        "Output tokens:",
        result["output_tokens"]
    )

    print(
        "Total tokens:",
        result["total_tokens"]
    )
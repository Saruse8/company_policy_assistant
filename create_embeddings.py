import csv
import os

import numpy as np
from dotenv import load_dotenv
from google import genai


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
# 3. Convert each policy to text
# --------------------------------------------------

def policy_to_text(policy):
    return (
        f"TITLE: {policy['title']}\n"
        f"DEPARTMENT: {policy['department']}\n"
        f"CATEGORY: {policy['category']}\n"
        f"POLICY TEXT: {policy['policy_text']}"
    )


policy_texts = [policy_to_text(policy) for policy in policies]


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

print(f"Creating embeddings for {len(policies)} policies...")

policy_embeddings = []

for number, text in enumerate(policy_texts, start=1):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values
    policy_embeddings.append(embedding)

    print(f"Created embedding {number}/{len(policies)}")


# --------------------------------------------------
# 5. Save embeddings locally
# --------------------------------------------------

policy_embeddings = np.array(policy_embeddings)

np.save("policy_embeddings.npy", policy_embeddings)

print()
print("Finished.")
print("Embeddings saved to policy_embeddings.npy")
print("Shape:", policy_embeddings.shape)
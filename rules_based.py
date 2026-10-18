import csv
import re
import time


# --------------------------------------------------
# 1. Load company policies
# --------------------------------------------------

with open("company_policies.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    policies = list(reader)


# --------------------------------------------------
# 2. Stop words
# --------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "i", "me", "my", "you", "your",
    "we", "our", "do", "does", "did", "is", "are", "am",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "from", "with", "and", "or", "but", "if",
    "how", "what", "when", "where", "who", "why",
    "can", "could", "should", "would", "need"
}


# --------------------------------------------------
# 3. Clean text
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def get_keywords(text):
    words = clean_text(text).split()

    return set(
        word for word in words
        if word not in STOP_WORDS
    )


# --------------------------------------------------
# 4. Rules-based policy search
# --------------------------------------------------

def search_policy(question):

    start_time = time.perf_counter()

    question_words = get_keywords(question)

    best_policy = None
    best_score = 0


    for policy in policies:

        searchable_text = (
            policy["title"] + " " +
            policy["category"] + " " +
            policy["policy_text"]
        )

        policy_words = get_keywords(searchable_text)

        score = len(
            question_words.intersection(policy_words)
        )


        if score > best_score:
            best_score = score
            best_policy = policy


    response_time = time.perf_counter() - start_time


    if best_policy is None:

        return {
            "answer": "No relevant policy found.",
            "policy": "None",
            "response_time": response_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }


    return {
        "answer": best_policy["policy_text"],
        "policy": best_policy["title"],
        "response_time": response_time,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }


# --------------------------------------------------
# 5. Run manually only when this file is executed
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a policy question: ")

    result = search_policy(question)

    print()
    print("Answer:", result["answer"])
    print("Relevant policy:", result["policy"])
    print(
        "Response time:",
        round(result["response_time"], 6),
        "seconds"
    )
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
    print("Total tokens:", result["total_tokens"])

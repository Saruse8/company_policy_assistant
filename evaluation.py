import csv
import time

from rules_based import search_policy
from llm_no_vector import ask_llm
from vector_rag import ask_vector_llm


# --------------------------------------------------
# 1. Final benchmark questions
# --------------------------------------------------

test_questions = [

    # DIRECT QUESTIONS

    {
        "question": "How many vacation days do employees receive per year?",
        "expected_policies": ["Vacation Policy"],
        "covered": True,
        "type": "Direct"
    },

    {
        "question": "How long is the probation period for new employees?",
        "expected_policies": ["Probation Period Policy"],
        "covered": True,
        "type": "Direct"
    },

    {
        "question": "How often do I need to change my password?",
        "expected_policies": ["Password Policy"],
        "covered": True,
        "type": "Direct"
    },

    {
        "question": "How many days of bereavement leave can an employee take?",
        "expected_policies": ["Bereavement Leave Policy"],
        "covered": True,
        "type": "Direct"
    },


    # PARAPHRASE QUESTIONS

    {
        "question": "How frequently must I update my login credentials?",
        "expected_policies": ["Password Policy"],
        "covered": True,
        "type": "Paraphrase"
    },

    {
        "question": "Can I do my job from home three days each week?",
        "expected_policies": ["Remote Work Policy"],
        "covered": True,
        "type": "Paraphrase"
    },

    {
        "question": "Am I allowed to work from another country for one month?",
        "expected_policies": ["Work From Abroad Policy"],
        "covered": True,
        "type": "Paraphrase"
    },

    {
        "question": "What should I do if I notice a possible cybersecurity incident?",
        "expected_policies": ["Security Incident Policy"],
        "covered": True,
        "type": "Paraphrase"
    },


    # REASONING / REALISTIC SCENARIOS

    {
        "question": "I want to spend $700 on a business expense. Do I need approval first?",
        "expected_policies": ["Expense Approval Policy"],
        "covered": True,
        "type": "Reasoning"
    },

    {
        "question": "I am leaving the company. What should I do with my company laptop?",
        "expected_policies": [
            "IT Asset Management Policy",
            "Company Property Policy"
        ],
        "covered": True,
        "type": "Reasoning"
    },

    {
        "question": "Can I install Spotify on my company laptop?",
        "expected_policies": [
            "Software Installation Policy",
            "Laptop Usage Policy"
        ],
        "covered": True,
        "type": "Reasoning"
    },

    {
        "question": "A journalist contacted me about the company. Am I allowed to speak to them?",
        "expected_policies": ["Media Contact Policy"],
        "covered": True,
        "type": "Reasoning"
    },


    # QUESTIONS NOT COVERED BY THE DATABASE

    {
        "question": "Does the company provide free lunch every Friday?",
        "expected_policies": ["None"],
        "covered": False,
        "type": "Unsupported"
    },

    {
        "question": "What percentage annual bonus do employees receive?",
        "expected_policies": ["None"],
        "covered": False,
        "type": "Unsupported"
    },

    {
        "question": "Can employees bring their dogs to the office?",
        "expected_policies": ["None"],
        "covered": False,
        "type": "Unsupported"
    }
]


# --------------------------------------------------
# 2. Methods
# --------------------------------------------------

methods = [

    {
        "name": "Rules-based",
        "function": search_policy
    },

    {
        "name": "LLM without vector index",
        "function": ask_llm
    },

    {
        "name": "LLM with vector index",
        "function": ask_vector_llm
    }
]


# --------------------------------------------------
# 3. Helper function for policy matching
# --------------------------------------------------

def policy_is_correct(predicted_policy, expected_policies):

    predicted = predicted_policy.strip().lower()

    for expected in expected_policies:

        expected_clean = expected.strip().lower()

        if expected_clean in predicted:
            return True

    return False


# --------------------------------------------------
# 4. Store results
# --------------------------------------------------

results = []


# --------------------------------------------------
# 5. Run benchmark
# --------------------------------------------------

for question_number, test in enumerate(
    test_questions,
    start=1
):

    question = test["question"]
    expected_policies = test["expected_policies"]
    covered = test["covered"]
    question_type = test["type"]


    print()
    print("=" * 70)

    print(
        f"QUESTION {question_number}/{len(test_questions)}"
    )

    print(
        "Type:",
        question_type
    )

    print(
        "Question:",
        question
    )

    print(
        "Expected:",
        ", ".join(expected_policies)
    )

    print("=" * 70)


    for method in methods:

        method_name = method["name"]
        function = method["function"]

        print()
        print("Running:", method_name)


        try:

            result = function(question)

            predicted_policy = result["policy"]

            correct = policy_is_correct(
                predicted_policy,
                expected_policies
            )


            # ------------------------------------------
            # Unsupported-answer behaviour
            #
            # For questions known to be absent from the
            # database, the correct behaviour is to
            # return "None".
            # ------------------------------------------

            unsupported_answer = False

            if not covered:

                if predicted_policy.strip().lower() != "none":
                    unsupported_answer = True


            row = {

                "question_number": question_number,

                "question_type": question_type,

                "question": question,

                "expected_policies":
                    " | ".join(expected_policies),

                "covered_by_database": covered,

                "method": method_name,

                "answer": result["answer"],

                "predicted_policy":
                    predicted_policy,

                "policy_correct":
                    correct,

                "unsupported_answer":
                    unsupported_answer,

                "response_time":
                    result["response_time"],

                "input_tokens":
                    result["input_tokens"],

                "output_tokens":
                    result["output_tokens"],

                "total_tokens":
                    result["total_tokens"]
            }


            results.append(row)


            print(
                "Predicted policy:",
                predicted_policy
            )

            print(
                "Policy correct:",
                correct
            )

            if not covered:

                print(
                    "Unsupported answer:",
                    unsupported_answer
                )


            print(
                "Response time:",
                round(
                    result["response_time"],
                    3
                ),
                "seconds"
            )

            print(
                "Total tokens:",
                result["total_tokens"]
            )


        except Exception as error:

            print(
                "ERROR:",
                str(error)
            )


            results.append({

                "question_number":
                    question_number,

                "question_type":
                    question_type,

                "question":
                    question,

                "expected_policies":
                    " | ".join(
                        expected_policies
                    ),

                "covered_by_database":
                    covered,

                "method":
                    method_name,

                "answer":
                    "ERROR",

                "predicted_policy":
                    "ERROR",

                "policy_correct":
                    False,

                "unsupported_answer":
                    False,

                "response_time":
                    0,

                "input_tokens":
                    0,

                "output_tokens":
                    0,

                "total_tokens":
                    0
            })


        # Small pause between API calls
        time.sleep(1)


# --------------------------------------------------
# 6. Save detailed results
# --------------------------------------------------

output_file = "final_evaluation_results.csv"


fieldnames = [

    "question_number",

    "question_type",

    "question",

    "expected_policies",

    "covered_by_database",

    "method",

    "answer",

    "predicted_policy",

    "policy_correct",

    "unsupported_answer",

    "response_time",

    "input_tokens",

    "output_tokens",

    "total_tokens"
]


with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# --------------------------------------------------
# 7. Final summary
# --------------------------------------------------

print()
print()

print("=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)


for method in methods:

    method_name = method["name"]


    method_results = [

        row

        for row in results

        if row["method"] == method_name
    ]


    # ----------------------------------------------
    # Policy accuracy
    # ----------------------------------------------

    correct_count = sum(

        row["policy_correct"]

        for row in method_results
    )


    policy_accuracy = (

        correct_count /

        len(method_results) *

        100
    )


    # ----------------------------------------------
    # Average response time
    # ----------------------------------------------

    average_time = sum(

        row["response_time"]

        for row in method_results

    ) / len(method_results)


    # ----------------------------------------------
    # Average token use
    # ----------------------------------------------

    average_tokens = sum(

        row["total_tokens"]

        for row in method_results

    ) / len(method_results)


    average_input_tokens = sum(

        row["input_tokens"]

        for row in method_results

    ) / len(method_results)


    # ----------------------------------------------
    # Unsupported-answer rate
    # ----------------------------------------------

    unsupported_questions = [

        row

        for row in method_results

        if row["covered_by_database"] is False
    ]


    unsupported_count = sum(

        row["unsupported_answer"]

        for row in unsupported_questions
    )


    unsupported_rate = (

        unsupported_count /

        len(unsupported_questions) *

        100
    )


    # ----------------------------------------------
    # Print summary
    # ----------------------------------------------

    print()
    print(method_name)

    print(
        "Policy accuracy:",
        f"{policy_accuracy:.1f}%"
    )

    print(
        "Average response time:",
        f"{average_time:.3f} seconds"
    )

    print(
        "Average input tokens:",
        f"{average_input_tokens:.1f}"
    )

    print(
        "Average total tokens:",
        f"{average_tokens:.1f}"
    )

    print(
        "Unsupported-answer rate:",
        f"{unsupported_rate:.1f}%"
    )


print()
print(
    "Detailed results saved to:",
    output_file
)
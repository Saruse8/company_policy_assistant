import streamlit as st
import pandas as pd

from rules_based import search_policy
from llm_no_vector import ask_llm
from vector_rag import ask_vector_llm


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }

    .method-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
        margin-bottom: 0.2rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Company Policy Assistant")

st.markdown(
    """
    <div class="subtitle">
    Compare three approaches to answering questions from the same
    company policy database.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Ask a policy question:",
    placeholder="e.g. How frequently must I update my login credentials?"
)

compare_button = st.button(
    "Compare approaches",
    type="primary"
)


# --------------------------------------------------
# Run comparison
# --------------------------------------------------

if compare_button:

    if not question.strip():

        st.warning("Please enter a policy question.")

    else:

        with st.spinner("Comparing the three approaches..."):

            rules_result = search_policy(question)
            llm_result = ask_llm(question)
            vector_result = ask_vector_llm(question)


        st.divider()

        st.header("Results")

        col1, col2, col3 = st.columns(
            3,
            gap="large"
        )


        # --------------------------------------------------
        # Rules-based
        # --------------------------------------------------

        with col1:

            st.markdown(
                '<div class="method-label">Approach 1</div>',
                unsafe_allow_html=True
            )

            st.subheader("Rules-based")

            st.write(
                rules_result["answer"]
            )

            st.markdown(
                f"**Relevant policy:** "
                f"{rules_result['policy']}"
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Response time",
                    f"{rules_result['response_time']:.3f} s"
                )

            with metric2:
                st.metric(
                    "Input tokens",
                    rules_result["input_tokens"]
                )

            st.metric(
                "Total tokens",
                rules_result["total_tokens"]
            )


        # --------------------------------------------------
        # LLM without vector index
        # --------------------------------------------------

        with col2:

            st.markdown(
                '<div class="method-label">Approach 2</div>',
                unsafe_allow_html=True
            )

            st.subheader("LLM without vector index")

            st.write(
                llm_result["answer"]
            )

            st.markdown(
                f"**Relevant policy:** "
                f"{llm_result['policy']}"
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Response time",
                    f"{llm_result['response_time']:.3f} s"
                )

            with metric2:
                st.metric(
                    "Input tokens",
                    llm_result["input_tokens"]
                )

            st.metric(
                "Total tokens",
                llm_result["total_tokens"]
            )


        # --------------------------------------------------
        # LLM with vector index
        # --------------------------------------------------

        with col3:

            st.markdown(
                '<div class="method-label">Approach 3</div>',
                unsafe_allow_html=True
            )

            st.subheader("LLM with vector index")

            st.write(
                vector_result["answer"]
            )

            st.markdown(
                f"**Relevant policy:** "
                f"{vector_result['policy']}"
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Response time",
                    f"{vector_result['response_time']:.3f} s"
                )

            with metric2:
                st.metric(
                    "Input tokens",
                    vector_result["input_tokens"]
                )

            st.metric(
                "Total tokens",
                vector_result["total_tokens"]
            )


        # --------------------------------------------------
        # Vector retrieval details
        # --------------------------------------------------

        st.write("")

        with st.expander(
            "How did the vector index retrieve the policies?"
        ):

            st.write(
                "The vector approach first retrieves the three "
                "policies that are semantically closest to the "
                "question. Only these policies are then provided "
                "to the LLM."
            )

            for item in vector_result["retrieved"]:

                st.markdown(
                    f"**{item['policy']['title']}**  \n"
                    f"Similarity score: "
                    f"{item['similarity']:.3f}"
                )


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

st.divider()

st.header("Benchmark comparison")

st.caption(
    "Results from a 15-question benchmark containing direct, "
    "paraphrased, reasoning, and unsupported questions."
)


benchmark_data = {

    "Approach": [
        "Rules-based",
        "LLM without vector index",
        "LLM with vector index"
    ],

    "Policy accuracy": [
        "53.3%",
        "100.0%",
        "100.0%"
    ],

    "Avg. response time": [
        "0.002 s",
        "10.066 s",
        "14.210 s"
    ],

    "Avg. input tokens": [
        "0",
        "2,886.4",
        "223.3"
    ],

    "Avg. total tokens": [
        "0",
        "2,921.1",
        "258.0"
    ],

    "Unsupported-answer rate": [
        "100.0%",
        "0.0%",
        "0.0%"
    ]
}

benchmark_df = pd.DataFrame(
    benchmark_data
)

st.dataframe(
    benchmark_df,
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# Exactly two comparison paragraphs
# --------------------------------------------------

st.write(
    """
The rules-based approach was by far the fastest and required no LLM tokens, but it achieved only 53.3% policy accuracy in the 15-question benchmark and returned an unsupported answer for all three questions that were not covered by the policy database. The LLM without a vector index achieved 100% policy accuracy and a 0% unsupported-answer rate on these test questions, but it processed an average of 2,886.4 input tokens per question because the complete policy database was included in every prompt.
"""
)

st.write(
    """
The LLM with a vector index also achieved 100% policy accuracy and a 0% unsupported-answer rate in the benchmark, while reducing average input token use for answer generation to 223.3 tokens, approximately 92% less than the LLM without a vector index. However, this comparison excludes the additional token usage associated with generating query embeddings. Although the vector-index approach had a higher average response time in this benchmark (14.210 seconds compared with 10.066 seconds), it is preferred because it maintained the same measured accuracy and avoided unsupported answers in the tested cases, while substantially reducing the amount of policy context sent to the LLM.
"""
)
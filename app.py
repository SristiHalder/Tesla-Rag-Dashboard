import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# =========================================================
# LOAD DOCUMENT CHUNKS (from GitHub repo files)
# =========================================================
chunks_df = pd.read_csv("chunks.csv")   # <-- must be in repo root or folder you reference

# =========================================================
# KPI DATA
# =========================================================
KPI_DATA = {
    "Annual Revenue (2024)": "$96B",
    "Net Income (2024)": "$15B",
    "Gross Margin": "18%",
    "Operating Margin": "8%",
    "Automotive Revenue": "$82B",
    "Energy Storage Revenue": "$8B",
    "Services Revenue": "$6B",
    "YoY Revenue Growth": "19%",
}

# =========================================================
# SEGMENT TABLE DATA
# =========================================================
segment_df = pd.DataFrame({
    "Segment": ["Automotive", "Energy Storage", "Services"],
    "Revenue ($B)": [82, 8, 6],
    "YoY Growth (%)": [12, 60, 10],
    "Operating Margin (%)": [17, 25, 3]
})


# =========================================================
# BUILD TF-IDF MATRIX
# =========================================================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(chunks_df["chunk_text"])

def retrieve_chunks(question, top_k=3):
    vec = vectorizer.transform([question])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1][:top_k]

    return [
        {
            "chunk_id": int(chunks_df.iloc[i]["chunk_id"]),
            "source": chunks_df.iloc[i]["source_file"],
            "similarity": float(sims[i]),
            "text": chunks_df.iloc[i]["chunk_text"],
        }
        for i in idxs
    ]


# =========================================================
# PROMPT BUILDER
# =========================================================
def build_prompt(question, retrieved):
    context = ""
    for c in retrieved:
        context += f"\n[CHUNK {c['chunk_id']} - {c['source']}]\n{c['text']}\n"

    return f"""
You are an AI answering questions about Tesla using ONLY the retrieved text.
Cite chunk numbers like this: (Chunk 2).

QUESTION:
{question}

RETRIEVED TEXT:
{context}

INSTRUCTIONS:
• Use ONLY the retrieved chunks.
• No outside knowledge.
• Be clear, concise, fully factual.
• Provide citations after each claim.

ANSWER:
"""


# =========================================================
# SECURE OPENAI CLIENT FOR STREAMLIT DEPLOYMENT
# =========================================================
def get_openai_client():
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets.")
        st.stop()

    return OpenAI(api_key=api_key)

client = get_openai_client()


def answer_question(question, retrieved):
    prompt = build_prompt(question, retrieved)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Tesla Intelligence Dashboard", layout="wide")

clean_theme = """
<style>

body, p, div, span, h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

div[data-testid="stExpander"] {
    background-color: #111111 !important;
    border-radius: 8px;
}

table {
    background-color: #cc0000 !important;
    border-radius: 6px;
}

thead th {
    background-color: #cc0000 !important;
    color: #ffffff !important;
    font-weight: bold;
    text-align: center !important;
}

tbody td {
    background-color: #cc0000 !important;
    color: #ffffff !important;
    text-align: center !important;
}

</style>
"""
st.markdown(clean_theme, unsafe_allow_html=True)


# =========================================================
# HEADERS + OVERVIEW
# =========================================================
st.title("Tesla Intelligence Agent — RAG Dashboard")

st.markdown("""
### Overview
This dashboard provides an interactive, retrieval-augmented analysis of Tesla using information extracted from its annual report, ESG disclosures, and external summaries. 
The system retrieves relevant evidence from document chunks and uses a language model to generate an answer constrained strictly to that evidence. 
Users can explore Tesla’s business segments, financial performance, sustainability goals, and major risks through KPIs, retrieved evidence, 
and citation-based answers.
""")


# =========================================================
# KPI SECTION
# =========================================================
st.subheader("Key Performance Indicators")
cols = st.columns(4)
for (label, value), col in zip(KPI_DATA.items(), cols * (len(KPI_DATA)//4 + 1)):
    with col:
        st.metric(label, value)


# =========================================================
# BUSINESS SEGMENT TABLE
# =========================================================
st.subheader("Tesla Business Segment Performance")
st.table(segment_df)


# =========================================================
# QUESTION INPUT
# =========================================================
st.subheader("Ask the Agent")

question_category = st.selectbox("Select category:", [
    "Company Snapshot", "Risk Factors", "ESG Priorities", "Financials", "Custom"
])

presets = {
    "Company Snapshot": [
        "What does Tesla do?",
        "What are Tesla’s business segments?",
        "How does Tesla generate revenue?",
    ],
    "Risk Factors": [
        "What major risks does Tesla identify?",
        "What competitive pressures affect Tesla?",
    ],
    "ESG Priorities": [
        "What sustainability goals does Tesla have?",
        "How does Tesla measure environmental impact?",
    ],
    "Financials": [
        "What is Tesla’s total revenue?",
        "What are Tesla’s margins?",
        "What are Tesla’s main revenue streams?",
    ],
}

if question_category == "Custom":
    user_question = st.text_input("Your Question:")
else:
    user_question = st.selectbox("Choose a question:", presets[question_category])


# =========================================================
# RUN RAG PIPELINE
# =========================================================
if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        retrieved = retrieve_chunks(user_question)

        st.subheader("Retrieved Evidence")
        for r in retrieved:
            with st.expander(f"Chunk {r['chunk_id']} — {r['source']} (score: {r['similarity']:.4f})"):
                st.write(r["text"])

        st.subheader("Final Answer")
        answer = answer_question(user_question, retrieved)
        st.success(answer)
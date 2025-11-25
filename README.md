# Chatbot with Sentiment, Emotion & Sarcasm Analysis  
### A Modular AI-Powered Conversational System  
**(Assignment: Tier-1 Completed ✔ | Tier-2 Completed ✔ | Enhancements Added ⭐)**

This project implements a conversational chatbot that performs both **conversation-level** and **message-level** sentiment analysis, fulfilling all requirements of the assignment.  
In addition, the system includes multiple advanced enhancements: sarcasm detection, emotion tagging, ensemble scoring, and a Streamlit UI.

---

# 🎯 Assignment Compliance Summary

## ✔ **Tier 1 — Mandatory (Completed)**  
- Maintains **full conversation history**  
- Generates **overall conversation sentiment** at the end  
- Indicates **overall emotional direction** (Positive / Negative / Neutral)  

## ✔ **Tier 2 — Additional Credit (Completed)**  
- Performs **sentiment evaluation per user message**  
- Displays the per-message sentiment output  
- Provides **trend analysis** of mood shifts across the full conversation  

## ⭐ Optional Enhancements (Bonus Credit)  
- Sarcasm detection (HF model + heuristics)  
- Emotion classification  
- Ensemble sentiment scoring (HF + VADER)  
- Streamlit web application  
- Export conversation transcript  
- Reset functionality  
- Clean modular structure with extendable components  

---

# 📦 Features Overview

### ✔ Conversation-Level Sentiment  
- Aggregates sentiment across all user messages  
- Supports mean & recency-weighted averaging  

### ✔ Message-Level Sentiment  
- neg / neu / pos / compound scores  
- Label classification (Positive, Negative, Neutral)  

### ✔ Sarcasm Detection (Advanced)  
- Uses Hugging Face sarcasm model if available  
- Falls back to heuristic sarcasm patterns  
- Adjusts sentiment (positive sarcasm → negative)  

### ✔ Emotion Recognition  
- Uses HF model: `j-hartmann/emotion-english-distilroberta-base`  
- Detects emotions like joy, sadness, anger, fear  

### ✔ Ensemble Sentiment (HF + VADER)  
- Blends transformer-based sentiment with VADER  
- Improves accuracy on nuanced text  

### ✔ Streamlit Web UI  
- Clean split view: Chat (left) + Analytics (right)  
- Per-message table with sentiment, emotion, sarcasm  
- Timeline plot of compound scores  
- Export transcript as JSON  
- Reset conversation button  

---

# 📁 Folder Structure

chatbot_sentiment_analysis/
│
├── src/
│ ├── chatbot.py # Chatbot logic
│ ├── sentiment.py # HF + VADER sentiment engine
│ ├── sarcasm.py # Sarcasm detector
│ ├── utils.py # Helper preprocessing + formatting
│ └── tests/ # (Optional) Pytest tests
│
├── webapp.py # Streamlit Web Application
├── main.py # CLI application entry point
├── requirements.txt
└── README.md


---

# 🛠 Technologies Used

- **Python 3.9+**
- **NLTK VADER** (sentiment analysis)  
- **Hugging Face Transformers** (sentiment, emotion, sarcasm)  
- **Streamlit** (web interface)  
- **Matplotlib** (sentiment timeline plotting)  
- **Regex + heuristic rules** (fallback sarcasm detection)  

---

# 🧠 Sentiment Logic Explained (As Required)

The sentiment analysis consists of three main components:

## 1. **Message-Level Scoring**
Each user message is scored using:

- HF Transformer model → POS/NEG/NEU probabilities  
- VADER → Lexical rule-based sentiment  

If **ensemble mode** is ON:
compound_final = (compound_hf + compound_vader) / 2


## 2. **Sarcasm Adjustment**
If sarcasm is detected and the message is originally positive:

new_compound = -abs(original_compound) * 0.75

This gently turns sarcastic praise into mild negativity.

## 3. **Conversation-Level Aggregation**

Two modes:

Mean (default):
overall = average(compound_scores)
Recency-weighted:
weight = 0.8^(distance_from_end)

## 4. **Trend Analysis**

Compares early and late conversation sentiment:

delta > 0.05 → Improving  
delta < -0.05 → Worsening  
else         → Stable



#▶️ How to Run the Project 

1. Install dependencies
pip install -r requirements.txt
2. Run the Command-Line Chatbot (CLI)
python src/main.py

You will see:
-Interactive conversation
-Per-message sentiment
-Overall sentiment
-Trend analysis

3. Run the Streamlit Web Application
streamlit run webapp.py

The web UI includes:
-Chat viewer
-Sentiment classification
-Emotion tags
-Sarcasm markers
-Trend graph
-Export function
-Reset button

🧪 Status of Tests (As Required)

Tests are optional per assignment — if tests are included, they cover:
-Sentiment scoring
-Sarcasm heuristic detection
-Conversation aggregation
-CLI behavior

To run tests:
pytest

⭐ Highlights of Innovations

This project goes beyond assignment requirements by adding:
-Sarcasm-aware sentiment correction
-Emotion classification using transformer models
-Ensemble sentiment scoring
-Streamlit analytics dashboard
-Exportable JSON transcripts
-Error-safe modular architecture
-Heuristics + ML hybrid design

# Chatbot with Sentiment, Emotion & Sarcasm Analysis  
### A Modular AI-Powered Conversational System (Tier-1 + Tier-2 Completed + Enhancements)

This project implements an interactive chatbot capable of:

- Maintaining full conversation history  
- Performing **conversation-level sentiment analysis**  
- Performing **per-message sentiment analysis**  
- Estimating **emotion labels** (optional HF model)  
- Detecting **sarcasm** (HF + heuristic hybrid)  
- Identifying sentiment **trends** across the conversation  
- Running via **Command Line Interface (CLI)**  
- Running via a full **Streamlit Web Application**

The system is designed to be modular, extensible, and production-friendly.

---

## 🚀 Features

### ✔ Tier 1 (Mandatory)
- Maintains full conversation history  
- Computes **overall conversation sentiment**  
- Outputs final aggregate sentiment (Mean / Recency Weighted)

---

### ✔ Tier 2 (Additional Credit)
- Sentiment evaluation for **every user message**  
- Displays per-message scores (neg, neu, pos, compound)  
- Summarizes **trend / mood shift** across conversation  

---

### ⭐ Additional Enhancements (Bonus Features)
These make the project stand out significantly:

#### 🔥 **Sarcasm Detection Module**
- Uses Hugging Face model (`mohitjain/sarcasm-roberta`) if available  
- Includes custom heuristic fallback  
- Adjusts sentiment when sarcasm is detected (positive → negative)  
- Visible in conversation + analysis dashboard

#### 🎭 **Emotion Classification (Optional)**
- Integrates with HF model:  
  `j-hartmann/emotion-english-distilroberta-base`  
- Adds emotion labels (e.g., joy, anger, sadness, fear)

#### 🔗 **Ensemble Sentiment Scoring**
- HF model + VADER are blended for improved robustness  
- Fixes the issue of HF giving false positives for subtle negativity

#### 📊 **Streamlit Web UI**
- Shows conversation on left, analytics on right  
- Table view of per-message scores  
- Trend plot (compound over time)  
- Export transcript as JSON  
- Reset conversation button

---

## 📁 Folder Structure

chatbot_sentiment_analysis/
│
├── src/
│ ├── chatbot.py # Chatbot logic
│ ├── sentiment.py # HF + VADER sentiment engine
│ ├── sarcasm.py # Sarcasm detector (HF + heuristic)
│ ├── utils.py # Preprocessing + formatting utilities
│ └── tests/ # Optional: pytest-based tests
│
├── webapp.py # Streamlit UI
├── main.py # CLI entrypoint
├── requirements.txt
└── README.md

---

## 🛠 Technologies Used

- **Python 3.9+**
- **NLTK VADER** sentiment analyzer  
- **Hugging Face Transformers** (`pipeline`)  
- **Streamlit** for UI  
- **Matplotlib** for graphs  
- **Regex-based heuristic sarcasm detection**

---

## 🧠 Sentiment Logic

### 1. **Token-level → Compound score**
- HF output (POS / NEG / NEU) is converted into compound [-1,1]
- VADER compound score is also computed
- When ensemble mode is ON → **average(compound_HF, compound_VADER)**

### 2. **Sarcasm Adjustment**
If sarcasm is detected **and original sentiment is positive**:

```new_compound = -abs(original_compound) * 0.75```

Meaning:
- Positive sarcastic messages become negative  
- Strength is moderately attenuated to avoid overcorrection  

### 3. **Conversation Aggregation**
Two strategies available:
mean — global average
recency — last messages weighted more heavily

### 4. **Trend Extraction**
Compare early-half sentiment vs late-half sentiment:
delta > 0.05 → Improving
delta < -0.05 → Worsening
else → Stable

---

## ▶ Running the Project

### **1. Install dependencies**
```bash
pip install -r requirements.txt

2. Run CLI Version
- python src/main.py

You will see:

- Chat interaction
- Final sentiment report
- Per-message results

3. Run Streamlit Web App

- streamlit run webapp.py

The UI includes:

- Chat window
- Sentiment scores
- Emotion labels
- Sarcasm indicator
- Adjustable model settings
- Reset + Export options
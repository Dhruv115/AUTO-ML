# AutoML Agent

An agentic AutoML pipeline built with **CrewAI** and **Groq (LLaMA 3.3-70B)**.
Automates the full ML workflow — EDA → preprocessing → model selection → tuning → reporting.

## Agents
- **EDA Agent** — loads CSV, runs exploratory analysis
- **Preprocessor Agent** — handles missing values, encoding, scaling
- **Model Agent** — selects and trains best ML model
- **Tuning Agent** — hyperparameter optimization
- **Report Agent** — generates final summary

## Stack
Python · CrewAI · LangChain · Groq · Scikit-learn

## Setup
```bash
pip install -r requirement.txt
# Add your GROQ_API_KEY to a .env file
# ADD CSV Path
python main.py
```

## Dataset
Tested on pain dataset (pain_dataset_200P_4hz.csv)

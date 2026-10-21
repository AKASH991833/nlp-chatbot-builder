---
title: BotForge
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# BotForge

Train-your-own intent chatbot (scikit-learn).

## Run locally
```bash
cd python-app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces
Create a new Space (SDK: streamlit), upload the contents of this folder, and it runs as-is -
`requirements.txt` and `app_file: app.py` are already set. No card or paid plan needed
(free CPU tier).

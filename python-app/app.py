"""BotForge - build, train and test your own intent-based chatbot (Streamlit).
Run: streamlit run app.py
No deep learning needed: TF-IDF + logistic regression trains in ~1 second.
"""
import json, os, pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="BotForge - NLP Chatbot Builder", page_icon="🤖", layout="wide")

@st.cache_data
def starter_intents():
    return json.load(open(os.path.join(HERE, "intents.json")))

if "intents" not in st.session_state:
    st.session_state.intents = starter_intents()
    st.session_state.model = None
    st.session_state.vec = None
    st.session_state.cv_score = None
    st.session_state.chat = []

st.title("🤖 BotForge")
st.caption("Design intents, add example phrases, train a real classifier (TF-IDF + logistic "
           "regression, scikit-learn), chat with your bot, then export it. No GPU, no API key - "
           "training takes about a second.")

intents = st.session_state.intents
n_ex = sum(len(v["examples"]) for v in intents.values())
c1, c2, c3 = st.columns(3)
c1.metric("Intents", len(intents))
c2.metric("Training examples", n_ex)
c3.metric("Model status", "trained ✅" if st.session_state.model else "not trained")

tab_data, tab_train, tab_chat, tab_export = st.tabs(["📚 Dataset", "🎯 Train", "💬 Chat", "📦 Export"])

with tab_data:
    st.subheader("Your intents")
    for name in sorted(intents):
        with st.expander(f"{name} - {len(intents[name]['examples'])} examples"):
            st.write("**Example phrases:**", ", ".join(intents[name]["examples"][:12]))
            st.write("**Bot replies:**", " · ".join(intents[name]["responses"]))
    st.divider()
    st.subheader("Add to the dataset")
    with st.form("add_intent"):
        new_intent = st.text_input("New intent name (e.g. appointment_booking)")
        new_examples = st.text_area("Example phrases, one per line (5+ recommended)")
        new_response = st.text_input("Bot reply for this intent")
        if st.form_submit_button("Add intent") and new_intent and new_examples and new_response:
            ex = [e.strip() for e in new_examples.splitlines() if e.strip()]
            if new_intent in intents:
                intents[new_intent]["examples"] += ex
                if new_response not in intents[new_intent]["responses"]:
                    intents[new_intent]["responses"].append(new_response)
            else:
                intents[new_intent] = {"examples": ex, "responses": [new_response]}
            st.session_state.model = None
            st.success(f"Intent '{new_intent}' saved with {len(ex)} examples. Retrain to use it.")
    with st.form("add_examples"):
        target = st.selectbox("...or add examples to an existing intent", sorted(intents))
        more = st.text_area("More example phrases, one per line")
        if st.form_submit_button("Add examples") and more.strip():
            ex = [e.strip() for e in more.splitlines() if e.strip()]
            intents[target]["examples"] += ex
            st.session_state.model = None
            st.success(f"Added {len(ex)} examples to '{target}'. Retrain to use them.")

with tab_train:
    st.subheader("Train the classifier")
    st.caption("5-fold cross-validation scores the model honestly before you trust it.")
    if st.button("Train now", type="primary"):
        X, y = [], []
        for name, v in intents.items():
            for e in v["examples"]:
                X.append(e); y.append(name)
        vec = (TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, sublinear_tf=True),
               TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True))
        V = hstack([vec[0].fit_transform(X), vec[1].fit_transform(X)])
        model = LogisticRegression(C=10, max_iter=2000, class_weight="balanced")
        folds = min(5, min(np.bincount(np.unique(y, return_inverse=True)[1])))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            st.session_state.cv_score = float(cross_val_score(model, V, y, cv=cv).mean())
        model.fit(V, y)
        st.session_state.model, st.session_state.vec = model, vec
        st.success("Trained!")
    if st.session_state.cv_score is not None:
        st.metric("Cross-validation accuracy", f"{st.session_state.cv_score*100:.0f}%")
        st.caption("Below ~80%? Add more varied examples per intent, or split overlapping intents.")

with tab_chat:
    if st.session_state.model is None:
        st.info("Train the model first (Train tab).")
    else:
        for u, b, intent, conf in st.session_state.chat:
            with st.chat_message("user"): st.write(u)
            with st.chat_message("assistant"):
                st.write(b)
                st.caption(f"intent: {intent} · confidence {conf:.0%}")
        if q := st.chat_input("Talk to your bot..."):
            V = hstack([st.session_state.vec[0].transform([q]),
                        st.session_state.vec[1].transform([q])])
            prob = st.session_state.model.predict_proba(V)[0]
            i = int(np.argmax(prob)); intent = st.session_state.model.classes_[i]
            conf = float(prob[i])
            if conf < 0.35:
                reply = "I'm not sure about that one - try rephrasing, or teach me this in the Dataset tab."
                intent = "fallback"
            else:
                reply = np.random.choice(intents[intent]["responses"])
            st.session_state.chat.append((q, reply, intent, conf))
            st.rerun()

with tab_export:
    st.subheader("Take your bot with you")
    st.download_button("Download intents.json", json.dumps(intents, indent=2),
                       "intents.json", "application/json")
    if st.session_state.model is not None:
        blob = pickle.dumps({"vectorizer": st.session_state.vec, "model": st.session_state.model,
                             "intents": intents})
        st.download_button("Download trained model (.pkl)", blob, "botforge_model.pkl",
                           "application/octet-stream")
        st.caption("Load with pickle: `d = pickle.load(open('botforge_model.pkl','rb'))`, then "
                   "`d['model'].predict(d['vectorizer'].transform(['hello']))`.")
    st.divider()
    st.subheader("Honest limits")
    st.caption("Intent bots answer what they were taught - they don't understand meaning the way "
               "a large language model does. Accuracy grows with more and more-varied examples "
               "(aim for 20+ per intent). Everything runs locally; your data never leaves the app.")

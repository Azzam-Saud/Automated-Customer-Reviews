import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = r"D:\Azzam\Personal_Projects\SDA\NLP\Automated Customer Reviews\model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping (you may adjust this according to your training)
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# UI
st.title("💬 Review Sentiment Classifier (RoBERTa Model)")
st.markdown("Classify a customer review into sentiment categories using a fine-tuned RoBERTa model.")

text = st.text_area("Enter a product review:", height=150) 

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter a review to classify.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
            pred = np.argmax(probs)

        st.success(f"**Predicted Sentiment: {label_map[pred]}**")
        st.markdown("#### Confidence Scores:")
        for i, label in label_map.items():
            st.write(f"- {label}: {probs[i]*100:.2f}%")

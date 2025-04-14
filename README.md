# 🛍️ Automated Customer Reviews Analysis

This project provides a Streamlit-powered web interface for two powerful NLP tasks:

1. **Review Sentiment Classification** using a fine-tuned RoBERTa model.
2. **Product Name Clustering** using TF-IDF vectorization and KMeans clustering.

---

## 📌 Project Structure


---

## 💡 Tasks Overview

### Task 1: Review Sentiment Classification

- Classifies reviews into:
  - **Positive**
  - **Neutral**
  - **Negative**

- Built using:
  - 🤗 Transformers (RoBERTa model fine-tuned on labeled customer reviews)
  - PyTorch for inference
  - Streamlit for the interactive UI

### Task 2: Product Name Clustering

- Groups similar product names into meta-categories such as:
  - `AmazonBasics Essentials`
  - `Kindle E-Readers`
  - `Alexa Devices & Accessories`
  - `Fire Tablets`
  - `Pet Supplies & Laptop Backpacks`

- Built using:
  - Scikit-learn's `TfidfVectorizer` for feature extraction
  - `KMeans` clustering to group similar products

---

## 🖥️ Running the App

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer-review-analysis.git
cd customer-review-analysis

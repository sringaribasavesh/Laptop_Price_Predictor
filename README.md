# 💻 Laptop Price & Purchase Recommendation System

An end-to-end machine learning project that predicts the fair price of a laptop based on its hardware specifications and evaluates real customer reviews to provide a smart *buy or not-buy* recommendation.  
Built using **Python, scikit-learn, NLP (TF-IDF + SVC), and Streamlit** for deployment.

---

## 🚀 Features

✔ Predicts laptop price based on key specifications  
✔ Extracts sentiment score from user-provided reviews using NLP  
✔ Generates both **quality verdict** and **deal verdict**  
✔ Interactive web UI powered by **Streamlit**  
✔ Inputs: Laptop specs, actual price, and sample reviews  
✔ Outputs:  
- Fair predicted price  
- Sentiment score (per review + average)  
- Final buying recommendation  

---

## 🧠 Machine Learning Workflow

### 🧱 1. Data Preprocessing & Feature Engineering
- Cleanup of dataset (removing rare values, standardizing fields)
- Feature extraction including:
  - Touchscreen & IPS boolean mapping  
  - **PPI (Pixels Per Inch)** calculation from resolution + screen size  
- One-Hot Encoding for categorical fields  
- Power and scaling transformations

### 🤖 2. Price Prediction Model
Several regression models were evaluated:

| Model | Result |
|-------|--------|
| Linear Regression | Baseline |
| Ridge / Lasso | Improved regularization |
| **Random Forest (Final)** | Best R² & Low MAE ✔ |

The final model was serialized using **Pickle**.

### 🗣️ 3. Sentiment Analysis (NLP)
- Text cleaned using token normalization and regex
- Vectorized using **TF-IDF**
- Trained a **Support Vector Classifier (SVC)** with probability calibration
- Produces:
  - Per-review positive sentiment probability
  - Aggregate sentiment score for quality assessment

---

## 🏗 Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| ML Framework | scikit-learn |
| NLP | TF-IDF + SVC |
| Deployment | Streamlit |
| Model Serialization | Pickle |
| Data | Laptop Specs Dataset + Sample Reviews |

---

## 🎮 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/laptop-price-predictor.git
cd laptop-price-predictor

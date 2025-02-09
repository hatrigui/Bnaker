# **Banker: Loan Prediction and Monitoring System**

## 📖 **Project Overview**  
**Banker** is a Big Data application designed to automate loan status predictions and provide clear visualizations for managers. The project integrates Machine Learning for predictions, Apache Spark for distributed data processing, and the ELK Stack (Elasticsearch, Logstash, Kibana) for storage and visualization of results.  

---

## 🚀 **Features**  
- **Loan Prediction**: Machine Learning models (XGBoost, CatBoost, Random Forest, etc.) predict whether a loan will be approved or rejected.  
- **Distributed Processing**: Handles large datasets using Apache Spark.  
- **Web Application**:  
  - **Frontend (Angular)**: Allows users to upload CSV files, select models, and generate predictions.  
  - **Backend (Flask API)**: Handles data preprocessing and prediction.  
- **Data Visualization**: Interactive dashboards in Kibana to analyze predictions, loan intents, credit scores, and more.  
- **Synthetic Data Generation**: Uses GANs (Generative Adversarial Networks) to enrich the dataset.  

---

## 🛠 **Technologies Used**  
- **Machine Learning**:  
  - Algorithms: XGBoost, CatBoost, Random Forest, Logistic Regression.  
  - Tools: Scikit-learn, Pandas, Joblib (for model persistence).  
- **Distributed Processing**: Apache Spark (deployed via Docker).  
- **ELK Stack**: Elasticsearch (storage), Logstash (data pipelines), Kibana (visualization).  
- **Web Development**:  
  - **Backend**: Flask (REST API), CORS.  
  - **Frontend**: Angular (components, routing).  
- **Containerization**: Docker and Docker Compose for Spark and ELK environments.  

---

## 📂 **Data Structure**  
- **Dataset**: 45,000 entries with 14 features, including:  
  - Age, gender, education level, income, credit history, etc.  
  - Target variable: `loan_status` (0 = rejected, 1 = approved).  
- **Preprocessing**:  
  - Encoding categorical variables (Label Encoding, One-Hot Encoding).  
  - Normalization with `RobustScaler`.  
  - Handling class imbalance via oversampling.  

---

## 📊 **Results**  
### **Model Performance:**  
| Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|--------|------------------|------------------|------------------|
| XGBoost | 0.89 | 0.80 | 0.84 |
| CatBoost | 0.89 | 0.80 | 0.84 |
| Random Forest | 0.90 | 0.78 | 0.83 |
| Logistic Regression | 0.77 | 0.74 | 0.76 |

### **Kibana Visualizations:**  
- **Prediction distribution**: 79% rejected, 21% approved.  
- **Most common loan intents**: EDUCATION, VENTURE.  
- **Correlations**: Between age, home ownership, and loan status.  



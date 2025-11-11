# 🚚 Amazon Delivery Time Prediction

A machine learning system that predicts delivery times for Amazon orders based on various factors like distance, traffic, weather conditions, and agent performance.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-orange)
![Web App](https://img.shields.io/badge/Web-App-Streamlit-green)

## 📊 Project Overview

This project addresses the challenge of predicting accurate delivery times for e-commerce orders by leveraging machine learning algorithms and real-time data analysis.

### 🎯 Key Features
- **Real-time Delivery Predictions**
- **Multiple Factor Analysis** (Distance, Traffic, Weather, etc.)
- **Machine Learning Model Comparison**
- **Interactive Web Interface**
- **Experiment Tracking with MLflow**

## 🏗️ Project Architecture
Data Collection → Data Preprocessing → Feature Engineering → Model Training → Web Deployment


## 📈 Model Performance

| Model | RMSE | MAE | R² Score | Status |
|-------|------|-----|----------|--------|
| Random Forest | 12.45 hours | 9.87 hours | 0.893 | ✅ Best |
| Gradient Boosting | 13.21 hours | 10.45 hours | 0.872 | |
| Linear Regression | 18.23 hours | 14.56 hours | 0.812 | |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SayyedMohsin/Amazon-Delivery-Time-Prediction.git
cd Amazon-Delivery-Time-Prediction

### Install dependencies

bash
pip install -r requirements_updated.txt
Prepare the data

bash
python fixed_data_preparation.py
Train the models

bash
python fixed_model_training.py
Launch the web application

bash
streamlit run perfect_streamlit_app.py
View MLflow experiments (Optional)

bash
mlflow ui

### 📁 Project Structure

Amazon-Delivery-Time-Prediction/
├── fixed_data_preparation.py     # Data cleaning & feature engineering
├── fixed_model_training.py       # Model training with MLflow
├── perfect_streamlit_app.py      # Streamlit web application
├── requirements_updated.txt      # Python dependencies
├── README.md                     # Project documentation
└── sample_data/                  # Sample dataset

### 🛠️ Technologies Used
Programming Language: Python 3.13

Machine Learning: Scikit-learn, Pandas, NumPy

Web Framework: Streamlit

Experiment Tracking: MLflow

Data Visualization: Matplotlib, Seaborn

Geospatial Analysis: Geopy

### 📊 Dataset Features
The model uses the following features for prediction:

Feature	Description	Type
Distance_km	Store to customer distance	Numerical
Agent_Age	Delivery agent age	Numerical
Agent_Rating	Agent performance rating	Numerical
Weather	Weather conditions	Categorical
Traffic	Traffic levels	Categorical
Vehicle	Delivery vehicle type	Categorical
Area	Delivery area type	Categorical
Order_Hour	Time of order	Numerical

### 🌐 Web Application
The Streamlit web application provides:

User-friendly interface for inputting order details

Real-time delivery time predictions

Smart insights and recommendations

Interactive visualizations

https://images/streamlit_app.png

### 📈 Results & Insights
Achieved 89.3% accuracy (R² Score) with Random Forest

Distance and Traffic are the most important factors

Model can predict delivery times within ±12.5 hours accuracy

Real-business impact for logistics optimization

### 🔮 Future Enhancements
Real-time traffic data integration

Live weather API integration

Mobile application development

Advanced deep learning models

Real-time package tracking

### 👨‍💻 Author
Sayyed Mohsin Ali

GitHub: @SayyedMohsin

Project Link: Amazon Delivery Time Prediction
# istanbul-rental-price-prediction
End-to-end Deep Learning (Keras) project predicting rental prices in Istanbul, comparing ANN with classical ML models.
# 🏠 Istanbul Rental Price Prediction with Deep Learning

## 📌 Project Overview
This project is an end-to-end Machine Learning and Deep Learning pipeline designed to predict rental house prices in Istanbul's dynamic real estate market. The primary objective is to evaluate and compare the performance of classical Machine Learning algorithms (Linear Regression, Random Forest) against a Deep Learning Artificial Neural Network (ANN) architecture.

*This project was developed as part of the "Mastering Data Science With Deep Learning" bootcamp led by instructor Zafer Acar.*

## 🚀 Features & Pipeline
- **Data Generation:** Simulates web-scraped real estate data encompassing key features like Districts, Square Meters, Room Counts, and Building Age.
- **Data Preprocessing:** Includes robust feature engineering, One-Hot Encoding for categorical variables, and Feature Scaling using `StandardScaler`.
- **Deep Learning Model:** A multi-layer `Sequential` ANN model built using Keras, optimized with the Adam optimizer.
- **Model Evaluation:** Visual and statistical comparison of Mean Absolute Error (MAE) and $R^2$ scores across different algorithms to prove the superiority of the deep learning approach.

## 🛠️ Technologies Used
- **Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn
- **Deep Learning:** TensorFlow, Keras
- **Data Visualization:** Matplotlib

## 📊 Results & Performance
The model was tested on unseen data (20% test split) to ensure it learned the underlying patterns rather than overfitting. The Deep Learning model significantly outperformed traditional methods:

- **Deep Learning (Keras) $R^2$ Score:** 0.9798
- **Deep Learning (Keras) MAE:** ~2101 TL

These metrics clearly demonstrate that the neural network successfully captured the non-linear relationships within the real estate data, making highly accurate, data-driven pricing predictions.

## 📂 Repository Structure
- `project_notebook.ipynb`: The main Jupyter Notebook containing the full end-to-end code.
- `istanbul_rental_houses.csv`: The generated dataset used for training and testing the models.
- `rental_prediction_model.keras`: The saved, fully trained Keras model ready for future predictions.

## 💡 How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed (`pip install pandas numpy scikit-learn tensorflow matplotlib`).
3. Run the Jupyter Notebook cell by cell or load the `.keras` model directly to make new predictions.

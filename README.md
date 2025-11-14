# Rainfall Prediction using Machine Learning

## Project Overview

This project focuses on predicting rainfall amounts using various machine learning regression models. The primary goal is to compare the performance of different algorithms in forecasting rainfall based on historical weather data and identify the most suitable model for this task. The project covers data preprocessing, model training, evaluation, and comparative analysis of multiple regression techniques.

## Dataset

The project utilizes a historical rainfall dataset (e.g., Ugandan rainfall data) that includes various meteorological features and corresponding rainfall measurements over time. The dataset undergoes thorough cleaning, feature engineering (e.g., extracting year, month, dekad from dates), and scaling to prepare it for model training.

## Methodology

1.  **Data Loading & Preprocessing**: The raw data is loaded, cleaned, missing values are handled, and new features are engineered. Categorical features are one-hot encoded, and numerical features are scaled using `MinMaxScaler`.
2.  **Data Splitting**: The dataset is divided into training and testing sets to evaluate model generalization capabilities.
3.  **Model Implementation & Training**: Six distinct machine learning regression models are implemented and trained on the preprocessed data:
    *   **Neural Network (NN)**
    *   **K-Nearest Neighbors (KNN)**
    *   **Support Vector Regressor (SVR)**
    *   **Gradient Boosting Regressor (GBR)**
    *   **Random Forest Regressor (RF)**
    *   **Decision Tree Regressor (DT)**
4.  **Model Evaluation**: Each model's performance is assessed using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
5.  **Comparative Analysis**: A detailed comparison of all models is conducted, highlighting their strengths and weaknesses.

## Key Findings and Results

After comprehensive evaluation, the **Decision Tree Regressor** and **Random Forest Regressor** consistently demonstrated superior performance, achieving significantly lower MAE and RMSE values compared to other models.

### Performance Summary:

*   **Decision Tree Regressor**: MAE = 0.43 mm, RMSE = 5.41 mm
*   **Random Forest Regressor**: MAE = 0.59 mm, RMSE = 3.03 mm
*   Neural Network (NN): MAE = 25.34 mm, RMSE = 34.64 mm
*   K-Nearest Neighbors (KNN): MAE = 26.03 mm, RMSE = 36.26 mm
*   Support Vector Regressor (SVR): MAE = 27.18 mm, RMSE = 38.41 mm
*   Gradient Boosting Regressor (GBR): MAE = 26.16 mm, RMSE = 35.65 mm

This indicates that tree-based models are highly effective for this particular rainfall prediction task, likely due to their ability to capture complex non-linear relationships within the dataset.

## Installation

To run this project, you will need the following Python libraries. You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd rainfall-prediction
    ```
2.  **Open the Jupyter/Colab Notebook**: The entire analysis and model implementation are contained within a single Jupyter/Colab notebook.
3.  **Run all cells**: Execute the cells sequentially to perform data loading, preprocessing, model training, and evaluation.

## Future Work

*   **Hyperparameter Tuning**: Optimize the hyperparameters of the best-performing models (Decision Tree and Random Forest) for even better accuracy.
*   **Advanced Feature Engineering**: Explore more sophisticated feature engineering techniques to extract additional predictive signals from the data.
*   **Time Series Models**: Investigate specialized time series models (e.g., ARIMA, LSTMs) to account for temporal dependencies more explicitly.
*   **Ensemble Methods**: Implement more advanced ensemble techniques or stacking to combine the strengths of multiple models.

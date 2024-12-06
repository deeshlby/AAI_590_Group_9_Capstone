# AAI_590_Group_9_Capstone
# Predictive Analytics for Type 1 Health (P.A.T.H.): An Integrated Diabetes Management System

## Project Overview

This project aims to develop an advanced predictive system for Type I diabetes management, leveraging the Tidepool dataset, DiaTrend dataset, and Nutrition 5k dataset. The system is divided into three major components: a personalized glucose prediction system using Continuous Glucose Monitoring (CGM) data, an HbA1c prediction model, and a computer vision-based model for meal assessment. The glucose prediction component utilizes LSTM models to forecast glucose levels, while the HbA1c component provides long-term health insights, and the meal assessment component estimates the nutritional content of meals from photographs. The ultimate goal is to enhance lifestyle control and improve health outcomes for individuals with Type I diabetes, providing insights into glycemic trends and meal-based insulin requirements.

## Data Sources

### Tidepool Dataset

The Tidepool dataset includes detailed continuous glucose monitoring (CGM) data and insulin pump events from multiple participants, providing a longitudinal view of glucose trends over time. This dataset includes:

- **Blood Glucose Measurements**: Regular CGM data points, representing glucose levels.
- **Basal and Bolus Insulin Doses**: Information on basal insulin delivery rates and bolus insulin administered to participants.
- **Patient Metadata**: Age, sex, and insulin-to-carbohydrate ratios for each participant, as well as event-specific details like physical activity.

### DiaTrend Dataset

The DiaTrend dataset, developed by Dartmouth College, provides longitudinal data for 54 patients with Type 1 diabetes. This dataset offers insights through:

- **Continuous Blood Glucose Levels**: Recorded over multiple months, used to track trends and predict HbA1c values.
- **Meal Events and Carbohydrate Intake**: Including associated glucose impact.
- **Patient-Specific Parameters**: Insulin sensitivity factors and demographic data.

### Nutrition 5k Dataset

This dataset contains over 5,000 images of meals along with their nutritional information, such as carbohydrates, proteins, and fats. These images are used to develop the meal assessment component based on CNNs and Vision Transformers, which helps estimate the glycemic impact of each meal.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Loading](#data-loading)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
   - [CGM Data Prediction Model](#cgm-data-prediction-model)
   - [Meal Assessment Model](#meal-assessment-model)
   - [HbA1c Prediction Model](#hba1c-prediction-model)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Deployment and Recommendations](#deployment-and-recommendations)
9. [Tools and Technologies](#tools-and-technologies)

## Environment Setup

To set up the project environment, install the necessary libraries using the provided setup scripts. Use the Jupyter notebooks within the repository for seamless local development, training, and evaluation.

## Data Loading

The data loading and exploration were handled through different notebooks, focusing on each dataset individually:

- **Tidepool Data Loading**: The `1_0_Exploratory_Data_Analysis_Tidepool_HCL.ipynb` notebook is used for data exploration and loading, covering initial exploration and visualization of Tidepool CGM data.
- **DiaTrend and Nutrition5k Loading**: The relevant notebooks also include data loading processes for DiaTrend and Nutrition5k, integrated directly into their respective EDA workflows.

## Exploratory Data Analysis

The exploratory data analysis (EDA) was performed across multiple notebooks:

- **Tidepool EDA (`1_0_Exploratory_Data_Analysis_Tidepool_HCL`&`1_1_Tidepool_EDA.ipynb`)**: Analysis of CGM glucose trends, patterns related to hypo- and hyperglycemia.
- **Nutrition5k EDA (`2_0_Nutritional_Values_Nutrition5k_EDA_Yolov8_Vision_Transformer.ipynb`)**: Summary of meal compositions, average macronutrients, and common ingredients.
- **DiaTrend EDA (`3_0_EDA_Diatrend.ipynb`)**: Insights on long-term glucose control, variability in HbA1c levels.


## Data Preprocessing

The preprocessing steps were implemented in several notebooks, including:

- **Normalization and Scaling**: Data was normalized to improve model performance across all datasets.
- **Data Augmentation**: Augmentation was applied to meal images to enhance the robustness of the Nutrition 5k model.
- **Feature Extraction**: Extracted key features such as meal carbohydrate content, glucose trend averages, and rolling statistics.

## Feature Engineering

- **Tidepool Data**: Engineered temporal features, such as glucose change rate and rolling mean, to provide insights into short-term glucose dynamics.
- **Nutrition 5k Data**: Generated key ingredients and excluded unimportant elements like salt and vinegar to focus on meal components that impact glycemic control.
- **DiaTrend Data**: Created patient-specific features including insulin-to-carbohydrate ratios and calculated derived metrics for HbA1c prediction.

## Model Development

### CGM Data Prediction Model
The `1_2_Feature_selection_and_Model_Tidepool_training_LSTM.ipynb` notebook documents the use of LSTM models to predict short-term glucose levels. Additionally, the `1_3_TidepoolHCL150_CGM_predict_High_vs_Low_LSTM.ipynb` focuses on distinguishing between high and low glucose events using classification techniques.

### Meal Assessment Model
The `2_1_nutrition5k_CNN_12-02.ipynb` notebook covers the development of a convolutional neural network (CNN) utlizing MobileNetV3 for analyzing meal images. `2_0_Nutritional_Values_Nutrition5k_EDA_Yolov8_Vision_Transformer.ipynb` includes experiments with Yolov8, and Vision Transformers for classifying meal components and estimating nutritional content.

### HbA1c Prediction Model
The `3_1_LSTM_A1C.ipynb` notebook details the development of the HbA1c prediction model using Bidirectional LSTM to estimate long-term glycemic control based on historical CGM data.

## Model Training and Evaluation

- **Metrics Used**: Mean Squared Error (MSE), Mean Absolute Error (MAE), Recall, Precision, and F1 Score.
- **Training Details**: Implemented hyperparameter tuning for each model, focusing on finding optimal learning rates, dropout rates, and minimizing overfitting.
- **Model Comparison**: LSTM and CNN models were evaluated based on their ability to predict outcomes such as glucose levels, HbA1c, and the glycemic impact of meals. Comparisons were made between the different architectures (e.g., Vision Transformers vs. CNNs).

## Deployment and Recommendations

- **Deployment**:While deployment was not part of this proejct, we reccoment a Model deployment strategy involveing AWS SageMaker for real-time inference of glucose trends and meal-based insulin recommendations.
- **User Integration**:A befit of going with Sagemaker would be that Data privacy and security are ensured using encrypted storage, compliant with HIPAA standards.
- **Recommendations**: Further personalization can be achieved by integrating patient-specific biomarkers for insulin sensitivity and individualized feedback to enhance predictions and outcomes.

## Tools and Technologies
This project uses a variety of tools and technologies, including:

- **Google Colab and VS Code**: For local and cloud-based notebook execution.
- **TensorFlow/Keras**: For developing deep learning models.
- **Pandas, NumPy, Scikit-learn**: For data preprocessing and manipulation.
- **Matplotlib, Seaborn**: For data visualization.

Future deployment recomendaitons

- **AWS SageMaker**: For model training and deployment.
- **AWS S3**: For data storage.

## Getting Started

1. Clone this repository.
2. Set up the environment by following the instructions in the environment setup notebook.
3. Follow the sequence of notebooks to load data, preprocess, develop models, and deploy.
4. Each notebook includes detailed instructions and documentation for easy navigation.

## Contributors
- Gary Takahashi
- Dina Shalaby
- Angel Benitez
- Eyoha Mengistu

## License
Apache License Version 2.0, January 2004
http://www.apache.org/licenses/

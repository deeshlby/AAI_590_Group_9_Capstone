# AAI_590_Group_9_Capstone
// Working template
# Type I Diabetes Management System with Advanced Predictive Modeling

## Project Overview

This project aims to develop an advanced predictive system for Type I diabetes management, leveraging the DiaTrend dataset and Nutrition 5k dataset. The system is divided into two major components: a personalized glucose prediction system using CGM data and a CNN-based model for meal assessment. The glucose prediction component utilizes LSTM/GRU models with attention mechanisms to forecast glucose levels, while the meal assessment component estimates the nutritional content of meals from photographs. The ultimate goal is to enhance lifestyle control and improve health outcomes for individuals with Type I diabetes, providing insights into glycemic trends and meal-based insulin requirements.

## Data Source
// TODO update with final list
### DiaTrend Dataset

This dataset consists of continuous glucose monitoring (CGM) and insulin pump data from 54 participants. It provides time series information collected every 5 minutes, including:

- **Blood Glucose Measurements**: Regular CGM data points, representing glucose levels.
- **Bolus Insulin Doses**: Records of insulin administered to participants.
- **Meal Announcements**: Information about meal events and estimated carbohydrate content.
- **Patient-Specific Parameters**: Insulin-to-carbohydrate ratios and hemoglobin A1c (HbA1c) levels.

### Nutrition 5k Dataset

This dataset contains images of various meals along with their nutritional content, such as carbohydrates, proteins, and fats. The data will be used to develop a meal assessment component based on computer vision techniques.

## Table of Contents
// TODO update based on final setup
1. [Environment Setup](#environment-setup)
2. [Data Loading](#data-loading)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
   - [CGM Data Prediction Model](#cgm-data-prediction-model)
   - [Meal Assessment Model](#meal-assessment-model)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Deployment and Recommendations](#deployment-and-recommendations)
9. [Tools and Technologies](#tools-and-technologies)

## Environment Setup

To set up the project environment:

1. Open the `0-Environment_Setup.ipynb` notebook.
2. Run all cells to install necessary libraries and set up the environment for local development, Colab, and AWS SageMaker.
3. Define global variables and configurations that will be used across the project.

## Data Loading

The `1-Load_Data.ipynb` notebook is used for data loading:

- Load data from local files or cloud storage (AWS S3).
- Provide options for loading DiaTrend data, Nutrition 5k data, and user-specific parameters.
- Perform basic sanity checks to ensure the data is properly formatted and ready for analysis.

## Exploratory Data Analysis

The `2-EDA.ipynb` notebook provides an introduction to the project and the dataset:

- Analyze glucose trends, meal impact on glucose levels, and insulin effectiveness.
- Explore patient-specific characteristics and their effect on glycemic control.
- Visualizations include time-series plots, carbohydrate analysis, and insulin response charts.

## Data Preprocessing

The `3-Data_Preprocessing.ipynb` notebook covers:

- Data cleaning, including handling missing values and removing outliers.
- Time series formatting and normalization.
- Extracting and engineering features for both the glucose prediction and meal assessment models.

## Feature Engineering

The `4-Feature_Engineering.ipynb` notebook includes:

- Creating patient-specific features, such as insulin sensitivity and carbohydrate ratios.
- Engineering meal-related features from the Nutrition 5k dataset.
- Identifying trends and seasonal components from the CGM data.

## Model Development

### CGM Data Prediction Model

The `5-Model_Development_CGM_Prediction.ipynb` notebook contains the development of a deep learning model using LSTM/GRU architectures with multi-head attention for glucose level prediction.

### Meal Assessment Model

The `6-Model_Development_CNN_Meal_Assessment.ipynb` notebook details the development of a CNN model for estimating nutritional content from meal images, with a focus on carbohydrate content and portion sizes.

## Model Training and Evaluation

The `7-Model_Training_and_Evaluation.ipynb` notebook:

- Details model training processes, including hyperparameter tuning.
- Evaluates performance using metrics such as Clarke Error Grid Analysis, Mean Absolute Error (MAE), and Time in Range (TIR).
- Compares predictions with real-world glucose responses.

## Deployment and Recommendations

The `8-Deployment_and_Recommendations.ipynb` notebook:

- Deploys the models to AWS SageMaker for real-time predictions.
- Discusses how the system provides personalized insulin recommendations based on meal assessment and glucose trends.
- Highlights safety features to prevent hypoglycemia and hyperglycemia.

## Tools and Technologies

This project uses a variety of tools and technologies, including:

- **AWS SageMaker**: For model training and deployment.
- **Google Colab and VS Code**: For local and cloud-based notebook execution.
- **TensorFlow/Keras**: For developing deep learning models.
- **AWS S3**: For data storage.
- **Pandas, NumPy, Scikit-learn**: For data preprocessing and manipulation.
- **Matplotlib, Seaborn**: For data visualization.

## Getting Started

1. Clone this repository.
2. Set up the environment by running the `0-Environment_Setup.ipynb` notebook.
3. Follow the notebooks in sequence to implement the data loading, preprocessing, model development, and deployment steps.
4. Each notebook includes detailed instructions and documentation.

## Contributors
//To do add links
- Gary
- Dian
- Team Member C

## License

Apache License Version 2.0, January 2004
http://www.apache.org/licenses/


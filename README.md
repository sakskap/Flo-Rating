# Predicting Chess Elo Ratings

This project aims to predict the Elo ratings of chess players based on the length of their games. Using machine learning models, we explore various approaches to accurately estimate the `WhiteElo` and `BlackElo` ratings from game data. This project includes data preprocessing, exploratory data analysis, model training, evaluation, and advanced model exploration.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Advanced Model Exploration](#advanced-model-exploration)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Usage](#usage)

## Overview

The goal of this project is to predict chess players' Elo ratings using machine learning techniques. We utilize features derived from chess game data to train and evaluate different regression models. The project is structured to provide insights into the data processing steps, model performance, and areas for improvement.

## Dataset

The dataset consists of chess game records with various features, including the moves made during the game and the Elo ratings of the players. The data was parsed from PGN (Portable Game Notation) files.

## Preprocessing

1. **Importing Necessary Libraries**: We use libraries such as `pandas`, `numpy`, and `sklearn` for data manipulation and modeling.
2. **Reading the Datasets**: The datasets are read into pandas DataFrames for easy manipulation.
3. **Checking for Missing Values**: Missing values are identified and handled appropriately.
4. **Removing Duplicate Rows**: Duplicate rows are removed to maintain data integrity.
5. **Parsing PGN Files**: PGN files are parsed to extract game data, including player moves and game metadata.
6. **Converting Elo Ratings to Numerical Data Types**: Elo ratings are converted to numerical types to enable numerical operations and statistical analysis.
7. **Feature Engineering**: New features, such as `MoveLength`, are created from the game data.

## Exploratory Data Analysis (EDA)

1. **Summary Statistics**: Summary statistics are generated to understand the distribution and central tendency of the data.
2. **Histograms**: Histograms are plotted for numerical columns to visualize their distributions.

## Model Training and Evaluation

1. **Data Splitting**: The dataset is split into training and test sets based on the availability of Elo ratings.
2. **Feature Scaling**: Features are scaled using `StandardScaler` to improve model performance.
3. **Training Models**: Different models, including `RandomForestRegressor` and `GradientBoostingRegressor`, are trained on the data.
4. **Model Evaluation**: The models are evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²).

## Advanced Model Exploration

We explore additional advanced models such as `XGBoost`, `LightGBM`, and `MLPRegressor` to improve prediction accuracy. The models are trained and evaluated in a similar manner to the initial models.

## Results

The results of the model evaluations are presented, highlighting the performance metrics for each model. Both Random Forest Regressor and Gradient Boosting Regressor had similar performance, but struggled to accurately predict the Elo ratings, as indicated by the negative R-squared values.

#### Gradient Boosting Regressor
- **WhiteElo**:
  - Mean Squared Error (MSE): 73,262.22
  - R-squared (R²): -0.0045
- **BlackElo**:
  - Mean Squared Error (MSE): 72,729.76
  - R-squared (R²): -0.0016

#### Random Forest Regressor
- **WhiteElo**:
  - Mean Squared Error (MSE): 73,520.49
  - R-squared (R²): -0.0080
- **BlackElo**:
  - Mean Squared Error (MSE): 72,811.55
  - R-squared (R²): -0.0027

## Conclusion

In this project, we aimed to predict the Elo ratings of chess players using various machine learning models. The results indicated that both models struggled to fit the data well, suggesting that the `MoveLength` feature alone may not be sufficient for accurate predictions. Future work should focus on feature engineering, model tuning, and exploring advanced models to improve predictive performance.

## Future Work

- **Feature Engineering**: Create additional features from the game data to provide more predictive power.
- **Model Tuning**: Conduct hyperparameter tuning for the models to optimize their performance.
- **Advanced Models**: Explore advanced models such as XGBoost, LightGBM, and neural networks.
- **Data Enrichment**: Include more data or enrich the existing dataset with additional relevant information.

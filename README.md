# PitchPredict

## Overview

This project is a ML pipeline that predicts the next pitch type that a pitcher will throw based on a sequence of previous pitches. The pipeline is built using the following steps:

1. [Data Collection](#data-collection)
2. [Data Cleaning](#data-cleaning)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)

## Data Collection

Data is sourced from Baseball Savant, a website that provides detailed baseball statistics. The data is collected using the pybaseball library, which is a Python wrapper for the Baseball Savant API. The returned data is pitch-by-pitch data for seasons 2015-2024, which is the entire timeframe that Baseball Savant has data for. This long timeframe is used to ensure that there is enough data to train the model and also test the model across multiple time periods, as the ideal model is trainable off the least amount of data possible. As pitchers arsenals and pitching strategies change over time, it is extremely desirable for a model to be accurate based off little data, allowing it to be used in real-time situations.

Documentation for the pybaseball library can be found [here](https://github.com/jldbc/pybaseball) and documentation for the Baseball Savant API can be found [here](https://baseballsavant.mlb.com/csv-docs).

## Data Cleaning

There is a very minimal amount of data cleaning applied to the data. In the future, the featurization and cleaning stages may be combined, but for now they are seperate. The clean stage primarily restricts the data to only include the pitchers and years of data which the model should be trained on. The cleaning stage also drops any duplicate rows in the data.

The source for the stage can be found at `src/clean.py`.

## Feature Engineering

The featurization stage is key for preparing the data to be used for training the resulting model. The stage takes the cleaned data as an input and creates the target variable, essentially just shifting the pitch type column up by one row. The full list of included features can be found at 'data/features.txt'. Currently most features available in the data are included, but this may be reduced in the future to only include the most important features. In particular I want to test how the model is able to perform with only situational features, as a lot of the other features (like batted ball result associated features) are not available across all pitches meaning that I have filled them with -1 values. Additionally, not all of this data may be available in real-time, so it is important to test how the model performs without it.

The source for the stage can be found at `src/featurize.py`.

## Model Training

The training stage is responsible for training the LSTM model on the prepared dataset. It is important to note that an individual model is trained for each pitcher, as these are highly specific applications and a general model cannot be expected to capture the nuances of each pitcher. This stage takes the featurized data as input and performs the following steps:

1. Data Preparation: The data is split by pitcher, and sequences of pitches are created for each pitcher. The features and target variable (next_pitch) are extracted, and the data is split into training, validation, and test sets. The features are scaled using MinMaxScaler.

2. Model Creation: An LSTM model is created using TensorFlow. The model consists of multiple LSTM layers with dropout and batch normalization, followed by a dense layer with a softmax activation function to perform the final multi-class classification.

3. Model Training: The model is compiled and trained using the training and validation sets. Early stopping and learning rate reduction callbacks are used to prevent overfitting and improve training efficiency.

4. Model Evaluation: The trained model is evaluated on the test set, and the test loss and accuracy are recorded. The model and its training history are saved for each pitcher.

5. Output: The trained models, along with their evaluation metrics and other relevant data, are saved to a pickle file for later use in the evaluation stage.

The source for the stage can be found at `src/train.py` and the model defionition can be found at `src/lstm_model.py`.

## Model Evaluation

The evaluation stage analyzes the performance of the trained LSTM models. This stage loads the trained models and their corresponding test data, performing the following evaluations:

1. Model Performance: The test loss and accuracy are calculated for each model, and the results are plotted to visualize the performance of the models.
2. Confusion Matrix: The confusion matrix is calculated for each model, and the results are plotted to visualize the distribution of predicted pitch types.

The source for the stage can be found at `src/evaluate.py`.
Plots can be found in the `data/outputs` directory.

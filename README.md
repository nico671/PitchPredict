# PitchPredict

## Overview

This project is a ML pipeline that predicts the next pitch type that a pitcher will throw based on a sequence of previous pitches. The pipeline is built using the following steps:

1. [Data Collection](#data-collection)
2. [Data Cleaning](#data-cleaning)
3. [Feature Engineering](#feature-engineering)
4. Model Training
5. Model Evaluation

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

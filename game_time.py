# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:06:57 2025

@author: jsaar
"""

#IMPORT PACKAGES AND DATA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

my_df = pd.read_csv("video_game_data.csv")

# SPLIT DATA INTO INPUT AND OUTPUT OBJECTS
X = my_df.drop(["completion_time"], axis = 1)
y = my_df["completion_time"]

#SPLIT DATA INTO TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#MODELLING
regressor = RandomForestRegressor()

#TRAIN THE MODEL
regressor.fit(X_train, y_train)

#ASSESS MODEL ACCURACY
y_pred = regressor.predict(X_test)

prediction_comparison = pd.DataFrame({"actual" : y_test, 
                                      "prediction": y_pred})

randforest_predict = r2_score(y_test, y_pred)



#Robustness Check
import numpy as np

# Load data
my_df = pd.read_csv("video_game_data.csv")
X = my_df.drop(["completion_time"], axis=1)
y = my_df["completion_time"]

# Store R² scores
r2_scores = []

# Repeat 10 times
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Final output
print("R² scores from 10 runs:", r2_scores)
loop_randforest_predict = ("Average R² score:", np.mean(r2_scores))

print(f"Average R2 score of loop: {loop_randforest_predict}, Average R2 score of one RandomForest: {randforest_predict}")

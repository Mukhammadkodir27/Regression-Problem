import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import pickle


# But we wish to have more metrics to evaluate the model
# and a single function that returns them all as a dictionary

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # return results in a form of a dictionary
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MedAE': medae,
        'MAPE': mape,
        'R2': r2
    }
    
    

# write a function that calculates various metrics for classification model, and allows 
# to define a custom threshold for the predicted probabilities and stores the results
# in a dictionary

def classification_metrics(y_true, y_pred_prob, cutoff = 0.5):
    y_pred = (y_pred_prob > cutoff).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision_1 = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    precision_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    recall_1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    recall_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    f1_score_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    # averages of the metrics for both classes [0, 1]
    balanced_accuracy = (recall_0 + recall_1) / 2
    balanced_precision = (precision_0 + precision_1) / 2
    balanced_f1 = (f1_score_0 + f1_score_1) / 2

    return {
        'AUROC' : roc_auc,
        'Accuracy' : accuracy,
        'Precision 1': precision_1,
        'Precision 0': precision_0,
        'Recall 1': recall_1,
        'Recall 0': recall_0,
        'F1 Score 1': f1_score_1,
        'F1 Score 0': f1_score_0,
        'Balanced Accuracy': balanced_accuracy,
        'Balanced precision': balanced_precision,
        'Balanced F1': balanced_f1
    }
    
    

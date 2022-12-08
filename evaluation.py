import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier

import support


def stability_metric(df):
    """
    Parameters
    ----------
    
    df : pd.DataFrame
    """
    df_use = df.copy()
    df_use = df_use.dropna(thresh = 2)
    df_use = df_use[[c for c in df_use.columns if "FI" in c]]
    
    errors = df_use.apply(lambda row: len(row.unique()) - 1, axis = 0)
    possibilities = (df_use.shape[0] - 1)*df_use.shape[1]
    
    metric = 1 - (errors/possibilities).sum()
    
    return np.round(metric, 3)
    
    

def lasso_benchmark(diz):
    
    benchmark = []
    for k, v in diz.items():
        
        sel_feats = [i for i in v if isinstance(i, dict)][0]
        acc_data = [i for i in v if isinstance(i, pd.DataFrame)
                    and "Accuracy" in i.columns][0]
        
        if isinstance(v[0], pd.DataFrame):
            small_X_train = v[0].iloc[:, sel_feats["Features"]]
        else:
            small_X_train = v[0][:, sel_feats["Features"]]
            
        if isinstance(v[1], pd.DataFrame):
            small_X_test = v[1].iloc[:, sel_feats["Features"]]
        else:
            small_X_test = v[1][:, sel_feats["Features"]]
        
        logreg = LogisticRegressionCV(Cs = 1/np.linspace(.1, 100, 300),
                                      penalty = "l1", n_jobs = -1,
                                      random_state = 42, solver = "saga",
                                      cv = 5, max_iter = 5e3, scoring = "accuracy")
        logreg.fit(small_X_train, v[2])

        mean_cv_scores = logreg.scores_[1].mean(axis = 0)

        # Lambda min (LogisticRegressionCV uses inverse of lambda)
        lambda_min = logreg.Cs_[np.argmax(mean_cv_scores)]

        # Lambda 1SE (LogisticRegressionCV uses inverse of lambda)
        one_se_min = np.max(mean_cv_scores) - (np.std(mean_cv_scores)/\
                                               np.sqrt(len(mean_cv_scores)))

        lambda_1se = logreg.Cs_[np.argmax(mean_cv_scores[mean_cv_scores <= one_se_min])]

        # Number of coefficients excluded at best lambda
        zero_coefs = np.sum(logreg.coef_ == 0)
        
        # Difference in accuracy
        benchmarked_best = acc_data.loc["Logistic Regression"]
        logreg_acc = logreg.score(small_X_test, v[3])

        benchmark += [[k, logreg, np.round(1/lambda_min, 3), np.round(1/lambda_1se, 3),
                       zero_coefs, np.round(logreg_acc-benchmarked_best, 3)]]
    return benchmark


def graphical_lasso_benchmark(benchmark):
    """
    Parameters
    ----------
    
    benchmark: list made of:
        - dataset name
        - sklearn.model
        - float
        - float
        - int
        - float
    """
    
    print("\n", benchmark[0])
    print()
    print(benchmark[1].coef_[0])

    lasso_fig, lasso_ax = plt.subplots(1, 2, figsize = (14, 4))

    lasso_ax[0].plot(1/(benchmark[1].Cs_),
                     benchmark[1].scores_[1].mean(axis = 0))

    lasso_ax[1].plot(benchmark[1].coefs_paths_[1].mean(axis = 0)[:, :-1],
                     label = range(len(benchmark[1].coef_[0]))) # exclude intercept
    lasso_ax[1].axhline(0, color = "black")
    lasso_ax[1].axvline(np.argmax(benchmark[1].scores_[1].mean(axis = 0)),
                        ls = "--", color = "black", label = "λ min")
    lasso_ax[1].legend()
    
    plt.show()
    
    
def just_benchmarking(X_train, X_test, y_train, y_test):
    
    # Logistic regression
    model1 = LogisticRegression(random_state = 42, n_jobs = -1,
                                max_iter = 5e3, solver = "saga")
    model1.fit(X_train, y_train)

    # Support vector machine
    model2 = LinearSVC(random_state = 42, max_iter = 1e4)
    model2.fit(X_train, y_train)

    # Random forest
    model3 = GradientBoostingClassifier(random_state = 42,
                                        min_samples_leaf = np.max([5,
                                                                   len(X_train)/100]).astype(int))
    model3.fit(X_train, y_train)

    # Neural network
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)

    model4 = support.build_MLP(X_train, y_train_cat,
                               {"Features":list(range(X_train.shape[1]))})


    # SCORES
    out = np.c_[[accuracy_score(y_test, model1.predict(X_test)),
                 accuracy_score(y_test, model2.predict(X_test)),
                 accuracy_score(y_test, model3.predict(X_test)),
                 accuracy_score(y_test, pd.DataFrame(model4.predict(X_test))\
                                        .idxmax(axis = 1).values )]]
    
    return out
    

def summary_table(data_dict, stability = True, neural_net = True):
    """
    Parameters
    ----------
    
    data_dict : {"dataset_name": [X_train,              --> pd.DataFrame / np.ndarray
                                  X_test,               --> pd.DataFrame / np.ndarray
                                  y_train,              --> pd.DataFrame / np.ndarray
                                  y_test,               --> pd.DataFrame / np.ndarray
                                  selected_variables,   --> iterable (list, np.array, ...) / dict
                                  accuracies,           --> iterable (list, np.array, ...)
                                  stability             --> iterable (list, np.array, ...)
                                  ]
                }
    
    stability : bool
    """
    summary = []
    for k, v in data_dict.items():

        # Dataset name
        # Sample size
        # Training size
        # Initial number of features
        # Selected features
        # Stability
        # ACCURACY: Full Logit
        # ACCURACY: Logistic Regression
        # ACCURACY: SVC
        # ACCURACY: Random Forest
        # ACCURACY: Neural Network
        # Average accuracy (full logit excluded)
        # Accuracy Standard deviation (full logit excluded)
        # Decreased accuracy (mean) w.r.t. full logit
        
        
        sel_features = v[4]["Features"] if isinstance(v[4], dict) else v[4]
        
        if stability == True:
            summary += [[k, len(v[2]) + len(v[3]), len(v[2]),
                         v[0].shape[1], len(sel_features),
                         v[6], *v[5]["Accuracy"].tolist(),
                         v[5]["Accuracy"][1:].mean(),
                         v[5]["Accuracy"][1:].std(),
                         v[5]["Accuracy"][1:].mean() - v[5]["Accuracy"].tolist()[0]]]

        else:
            summary += [[k, len(v[2]) + len(v[3]), len(v[2]),
                         v[0].shape[1], len(sel_features),
                         *v[5]["Accuracy"].tolist(),
                         v[5]["Accuracy"][1:].mean(),
                         v[5]["Accuracy"][1:].std(),
                         v[5]["Accuracy"][1:].mean() - v[5]["Accuracy"].tolist()[0]]]
            
    
    if stability == False:
        summ_columns = ["Dataset name", "Sample size", "Training size", "Number of Features",
                        "Selected Features", "Accuracy - Logit all features",
                        "Accuracy - Logit", "Accuracy - SVM", "Accuracy - RF",
                        "Accuracy - MLP", "Avg Acc. on selected features",
                        "Accuracy Std on selected features", "Acc. diff. wrt Full logit"]
        if neural_net == False:
            summ_columns = ["Dataset name", "Sample size", "Training size", "Number of Features",
                            "Selected Features", "Accuracy - Logit all features",
                            "Accuracy - Logit", "Accuracy - SVM", "Accuracy - RF",
                            "Avg Acc. on selected features",
                            "Accuracy Std on selected features", "Acc. diff. wrt Full logit"]
    else:
        summ_columns = ["Dataset name", "Sample size", "Training size", "Number of Features",
                        "Selected Features", "Feat. Top5-Stability", "Accuracy - Logit all features",
                        "Accuracy - Logit", "Accuracy - SVM", "Accuracy - RF",
                        "Accuracy - MLP", "Avg Acc. on selected features",
                        "Accuracy Std on selected features", "Acc. diff. wrt Full logit"]
    
    summary = pd.DataFrame(summary, columns = summ_columns)

    return summary.applymap(lambda v: v if isinstance(v, str) else round(v, 3))
    
    
    
    
    
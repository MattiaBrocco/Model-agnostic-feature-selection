import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV


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
    
    
    
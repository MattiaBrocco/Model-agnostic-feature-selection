import shap
import keras
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.random_seed import set_random_seed


def likelihood_ratio_test(model_red, model_full, X_train_red, X_train_full, y_train):
    """
    The likelihood-ratio test assesses the goodness of fit
    of two competing statistical models based on the ratio
    of their likelihoods, specifically one found by
    maximization over the entire parameter space and
    another found after imposing some constraint.
    
    Parameters
    ----------
    
    model1: 'sklearn.model' object (reduced)
    model2: 'sklearn.model' object (full)
    
    X_train1: 'pd.DataFrame'. Training set for reduced model.
    X_train2: 'pd.DataFrame'. Training set for full model.
    y_train: 'pd.DataFrame'. Common to model1 and model2.
    """
    
    ll_model_red = log_loss(y_train, model_red.predict(X_train_red))
    ll_model_full = log_loss(y_train, model_full.predict(X_train_full))
    
    lambda_stat = -2*(ll_model_red - ll_model_full)
    
    # H0: the two models have similar likelihood.
    # The likelihood-ratio test rejects H0
    # if the value of this statistic is too small.
    
    def_freedom = X_train_full.shape[1] - X_train_red.shape[1]
    
    # The numerator corresponds to the likelihood of an observed outcome under H0.
    # The denominator corresponds to the MAX log-likelihood of an observed outcome,
    # varying parameters over the whole parameter space.
    # The numerator of this ratio is less than the denominator; so, LLE in [0, 1].
    # [!!!] Low values of the likelihood ratio mean that the observed result
    # was much less likely to occur under the null hypothesis as compared
    # to the alternative. High values of the statistic mean that the observed
    # outcome was nearly as likely to occur under the null hypothesis as
    # the alternative, and so the null hypothesis cannot be rejected.
    
    p_value = np.float64(stats.chi2.sf(lambda_stat, def_freedom))
    
    return lambda_stat, p_value
    
    
def D3_pruning(X_train, y_train):
    
    tree = DecisionTreeClassifier(random_state = 42, criterion = "entropy")
    tree.fit(X_train, y_train)

    # 1.1 Pruning
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # 1.2 Cross-validation for parameters tuning
    tree_params = {"ccp_alpha":[ccp_alphas[impurities < impty].max()
                                for impty in np.linspace(max(0.1, impurities.min() + 0.01),
                                                         np.round(impurities.max()/2, 3),
                                                         num = 10)]}

    tree_grid = GridSearchCV(DecisionTreeClassifier(random_state = 42, criterion = "entropy"),
                             tree_params, cv = 3, n_jobs = -1, verbose = 0,
                             return_train_score = True)
    tree_grid.fit(X_train, y_train)
    
    return tree_grid


def sorted_importance_index(model, X, y, features):
    """
    Parameters
    ----------
    
    model : from sklearn or keras
    X, y : pd.DataFrame / np.array - X is the reduced dataset
    features : list
    """
    if isinstance(model, keras.engine.sequential.Sequential):
        
        explainer = shap.KernelExplainer(model, X[:100]) # first 100 rows
        shaps = explainer.shap_values(X[:100], nsamples = 30)

        abs_avg_shaps = np.abs(shaps[0]).mean(axis = 0) + np.abs(shaps[1]).mean(axis = 0)
        
        arr = np.c_[features, abs_avg_shaps]
    else:
        imps = permutation_importance(model, X, y, scoring = "accuracy",
                                      random_state = 42, n_repeats = 30,
                                      max_samples = min(len(X), 1000),
                                      n_jobs = -1)

        arr = np.c_[features, imps["importances_mean"]]

    
    out = arr[arr[:, 1].argsort()[::-1]][:, 0]
    
    return out.astype(int)[:5]


def scores_table(X_full, X_reduced):
    
    # Sample size
    # Number of features selected over the total number of features
    print("Train size: {}\nSelected {} features out of {}".format(len(X_full),
                                                                  X_reduced.shape[1],
                                                                  X_full.shape[1]))
    
def build_MLP(X_train, y_train_cat, features):
    """
    Parameters
    ----------

    X_train: output of 'prepare_data'
    y_train_cat: 'np.ndarray'
    features : 'dict'. output of 'variable_selection'
    """
    
    sel_features = features["Features"] if isinstance(features, dict) else features
    
    set_random_seed(101)    
    early_stopping = EarlyStopping(monitor = "val_loss", mode = "min",
                                   patience = 5, verbose = 0)

    model = Sequential(name = "ANN")
    model.add(Dense(input_dim = len(sel_features),
                    units = int(X_train.shape[1])/2,
                    activation = "relu"))
    model.add(Dense(units = 10, activation = "relu"))
    model.add(Dense(units = 2, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "adam",
                  metrics = ["accuracy"])
    
    if isinstance(X_train, pd.DataFrame):
        small_X_train = X_train.iloc[:, sel_features]
    else:
        small_X_train = X_train[:, sel_features]
    
    history = model.fit(small_X_train, y_train_cat,
                        epochs = 500, batch_size = min(len(X_train)/3, 50),
                        verbose = 0, validation_split = 0.3, callbacks = [early_stopping])
    
    return model
    
    
def data_to_feather(data_dir):
    """
    Parameters
    ----------

    data_dir: str. directory in which data is located
    
    Returns
    -------

    None
    But a feather file with the same name is written in 'data_dir'
    """
    
    if ".xlsx" in data_dir:
        data_import = pd.read_excel(data_dir)
    else:
        data_import = pd.read_csv(data_dir)

    if data_import.shape[1] == 1:
        df = pd.DataFrame([list(data_import.applymap(lambda s: s.split(";")).values)[i][0]
                           for i in range(len(data_import))])
        df.columns = [new_c.strip() for new_c in
                      data_import.columns[0].split(";")]

        for c in df.columns:
            try:
                df[c] = df[c].astype(int)
            except:
                continue

    else:
        df = data_import.copy()
        
    filename = data_dir.split("\\")[-1].split(".")[0]
    
    
    df.to_feather("{}\\{}.feather".format("\\".join(data_dir.split("\\")[:-1]), filename))
    
    return None


"""
APPENDIX

MERGE OF DATASETS FOR **R_NEO_PI**
```python
a = pd.read_excel(data_dir + "\\R_NEO_PI_Faked.xlsx")
b = pd.read_excel(data_dir + "\\R_NEO_PI_Honest.xlsx")

a.columns = [" ".join([pd.Series(a.columns).apply(lambda s: np.nan if "Unnamed"
                                                  in s else s).fillna(method = "ffill").tolist()[i],
                       a.loc[0][i]]) for i in range(len(a.columns))]
b.columns = [" ".join([pd.Series(b.columns).apply(lambda s: np.nan if "Unnamed"
                                                  in s else s).fillna(method = "ffill").tolist()[i],
                       b.loc[0][i]]) for i in range(len(b.columns))]

a = a.drop(0).reset_index(drop = True)
b = b.drop(0).reset_index(drop = True)

a["CONDITION"] = "FAKE"
b["CONDITION"] = "HONEST"

pd.concat([a, b], ignore_index = True).to_excel(data_dir + "\\R_NEO_PI.xlsx", index = False)
```
"""
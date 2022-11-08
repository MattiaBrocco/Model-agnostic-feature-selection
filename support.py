import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.random_seed import set_random_seed


def likelihood_ratio_test(model1, model2, X_train1, X_train2, y_train):
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
    ll_model1 = log_loss(y_train, model1.predict(X_train1))
    ll_model2 = log_loss(y_train, model2.predict(X_train2))
    
    lambda_stat = -2*(ll_model1 - ll_model2)
    
    # H0: the two models have similar likelihood.
    # The likelihood-ratio test rejects H0
    # if the value of this statistic is too small.
    
    def_freedom = X_train2.shape[1] - X_train1.shape[1]
    
    # The numerator corresponds to the likelihood of an observed outcome under H0.
    # The denominator corresponds to the MAX log-likelihood of an observed outcome,
    # varying parameters over the whole parameter space.
    # The numerator of this ratio is less than the denominator; so, LLE in [0, 1].
    # [!!!] Low values of the likelihood ratio mean that the observed result
    # was much less likely to occur under the null hypothesis as compared
    # to the alternative. High values of the statistic mean that the observed
    # outcome was nearly as likely to occur under the null hypothesis as
    # the alternative, and so the null hypothesis cannot be rejected.
    
    p_value = np.float64(1 - stats.chi2.cdf(lambda_stat, def_freedom))
    
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


def scores_table(X_full, X_reduced):
    
    # Sample size
    # Number of features selected over the total number of features
    print("Sample size: {}\nSelected {} features out of {}".format(len(X_full),
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
    
    set_random_seed(101)    
    early_stopping = EarlyStopping(monitor = "val_loss", mode = "min",
                                   patience = 5, verbose = 0)

    model = Sequential(name = "ANN")
    model.add(Dense(input_dim = len(features["Features"]),
                    units = int(X_train.shape[1])/2,
                    activation = "relu"))
    model.add(Dense(units = 10, activation = "relu"))
    model.add(Dense(units = 2, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "sgd",
                  metrics = ["accuracy"])
    
    history = model.fit(X_train[:, features["Features"]], y_train_cat,
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
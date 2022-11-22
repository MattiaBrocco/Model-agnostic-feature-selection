import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

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
    
    if isinstance(X_train, pd.DataFrame):
        small_X_train = X_train.iloc[:, features["Features"]]
    else:
        small_X_train = X_train[:, features["Features"]]
    
    history = model.fit(small_X_train, y_train_cat,
                        epochs = 500, batch_size = min(len(X_train)/3, 50),
                        verbose = 0, validation_split = 0.3, callbacks = [early_stopping])
    
    return model
    


def lasso_benchmark(diz):
    
    benchmark = []
    for k, v in diz.items():
        
        if isinstance(v[0], pd.DataFrame):
            small_X_train = v[0].iloc[:, v[-2]["Features"]]
        else:
            small_X_train = v[0][:, v[-2]["Features"]]
            
        if isinstance(v[1], pd.DataFrame):
            small_X_test = v[1].iloc[:, v[-2]["Features"]]
        else:
            small_X_test = v[1][:, v[-2]["Features"]]
        
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
        benchmarked_best = v[-1]["Logistic Regression"]
        logreg_acc = logreg.score(small_X_test, v[3])

        benchmark += [[k, logreg, np.round(1/lambda_min, 3), np.round(1/lambda_1se, 3),
                       zero_coefs, np.round(logreg_acc-benchmarked_best, 3)]]
    return benchmark
    
    
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
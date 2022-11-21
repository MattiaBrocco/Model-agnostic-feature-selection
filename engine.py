import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow.keras.utils import to_categorical

from joblib import delayed
from joblib import Parallel

import support


class Classification:
    
    def prepare_data(self, data_dir, target):
        """
        Parameters
        ----------
        
        data_dir: str. directory in which data is located
        target : str. columns in 'data' that contains the
                 values onto which prediction is made.
        
        Returns
        -------
        
        X_train, X_test, y_train, y_test
        """
        self.target = target
        self.data_dir = data_dir
        
        if ".xlsx" in data_dir:
            data_import = pd.read_excel(data_dir)
        elif ".feather" in data_dir:
            data_import = pd.read_feather(data_dir)
        elif ".parquet" in data_dir:
            data_import = pd.read_parquet(data_dir)
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
        
        # Perform train-test split
        X = df.drop(target, axis = 1)
        y = df[target]
        y = y.apply(lambda v: 1 if v == y.unique()[0] else 0) 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7,
                                                            random_state = 42, shuffle = True)
        # Scale data
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
         
        return X_train, X_test, y_train, y_test
    
    
    def variable_selection(self, X_train, X_test, y_train, y_test):
        """
        Parameters
        ----------
        
        X_test : 'pandas.DataFrame' or 'numpy.ndarray'
        y_test : 'pandas.DataFrame' or 'numpy.ndarray'
        X_train : 'pandas.DataFrame' or 'numpy.ndarray'
        y_train : 'pandas.DataFrame' or 'numpy.ndarray'
        """
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        # STEP 1: Permutation importance with a random forest.
        #         The forest inherits the complexity of a post-pruned tree.
        #         The tree is subjected to post-pruning (with CV)
        
        # 1.1 Pruning        
        pruned_tree = support.D3_pruning(X_train, y_train)
        # !!!!!!
        random_forest = GradientBoostingClassifier(random_state = 42,
                                                   ccp_alpha = pruned_tree.best_params_["ccp_alpha"])
        random_forest.fit(X_train, y_train)
        
        # 1.3 Feature selection
        perm_imp = permutation_importance(random_forest, X_test, y_test, n_repeats = 50,
                                          random_state = 42, scoring = "accuracy",
                                          n_jobs = -1)
        
        selected_features = pd.DataFrame(perm_imp["importances"].T)\
                            .apply(lambda col: 1 if
                                   stats.ttest_1samp(col, popmean = 0,
                                                     alternative = "greater")[1] < 0.001
                                   else 0)
        selected_features = np.where(selected_features != 0)[0].tolist()
        
        if isinstance(X_train, pd.DataFrame):
            small_X_train = X_train.iloc[:,selected_features]
        else:
            small_X_train = X_train[:,selected_features]
        
        #selected_features = []        
        #for i in perm_imp.importances_mean.argsort()[::-1]:
        #    if perm_imp.importances_mean[i] - 2 * perm_imp.importances_std[i] > 0:
        #        
        #        selected_features += [i]
        
        #if len(selected_features) < 2:
        #    selected_features = np.where(np.abs(perm_imp.importances_mean) > 1e-2)[0]
        #    selected_features = selected_features.tolist()
        #    if len(selected_features) < 2:
        #        selected_features = np.where(perm_imp.importances_mean != 0)[0]
        #        selected_features = selected_features.tolist()
        #else:
        #    pass
        #        #print("{} {:.3f} ± {:.3f}".format(X.columns[i],
        #        #                                  r2.importances_mean[i],
        #        #                                  r2.importances_std[i]))
                
        # STEP 2: Validation of feature selection. This is made
        #         By running a (generalized) linear model with
        #         all the features and the selected features.
        #         Then with a statistical test, if the full model
        #         is no better than the restricted model the validation is done.
        ######logreg_red = LogisticRegressionCV(Cs = 50, penalty = "l1", random_state = 42,
        ######                                  n_jobs = -1, solver = "saga", max_iter = 5e3)        
        logreg_red = LogisticRegression(n_jobs = -1, random_state = 42, max_iter = 5e3)
        logreg_red.fit(small_X_train, y_train)
        
        ######logreg_full = LogisticRegressionCV(Cs = 50, penalty = "l1", random_state = 42,
        ######                                   n_jobs = -1, solver = "saga", max_iter = 5e3)
        logreg_full = LogisticRegression(n_jobs = -1, random_state = 42, max_iter = 5e3)
        logreg_full.fit(X_train, y_train)
        
        # 2.1 Run test
        validation_test = support.likelihood_ratio_test(logreg_red, logreg_full,
                                                        small_X_train,
                                                        X_train, y_train) 
        # 2.2 Consider factor loading
        data_fl = pd.DataFrame(np.c_[small_X_train, y_train]).corr()
        fl = np.abs(data_fl)[len(selected_features)].min()
        
        # validation_test[1]: p-value of likelihood ratio test
        valid_lrt = True if validation_test[1] > 0.05 else False
        # Correlations
        valid_corr = True if fl >= 0.7 else False
        
        if valid_lrt is False:
            print("Warning: likelihood ratio test H0 should be rejected")
            
        return {"Features": selected_features,
                "Wilks test p-value": np.round(validation_test[1], 7),
                "High correlation": valid_corr}
    
    def benchmark_models(self, X_train, X_test, y_train, y_test, features):
        """
        Parameters
        ----------
        
        features : 'dict'. output of 'variable_selection'
        X_train, X_test, y_train, y_test : output of 'prepare_data'
        """
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.features = features
        
        if isinstance(X_train, pd.DataFrame):
            small_X_train = X_train.iloc[:, features["Features"]]
        else:
            small_X_train = X_train[:, features["Features"]]
            
        if isinstance(X_test, pd.DataFrame):
            small_X_test = X_test.iloc[:, features["Features"]]
        else:
            small_X_test = X_test[:, features["Features"]]
        
        
        # Full Logit
        model0 = LogisticRegression(random_state = 42, n_jobs = -1)
        model0.fit(X_train, y_train)
        
        # Logistic regression
        model1 = LogisticRegression(random_state = 42, n_jobs = -1)
        #model1.fit(X_train[:, features["Features"]], y_train)
        
        # Support vector machine
        model2 = SVC()
        #model2.fit(X_train[:, features["Features"]], y_train)
        
        # Random forest
        model3 = GradientBoostingClassifier(random_state = 42,
                                            min_samples_leaf = np.max([5,
                                                                       len(X_train)/100]).astype(int))
        #model3.fit(X_train[:, features["Features"]], y_train)
        
        
        
        fitted_models = Parallel(n_jobs = 8)(delayed(lambda m:
                                                     m.fit(small_X_train, y_train))(model)
                                             for model in [model1, model2, model3])
        
        # Neural network
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        model4 = support.build_MLP(X_train, y_train_cat, features)
        
        # SCORES
        out = pd.Series( [accuracy_score(y_test, model0.predict(X_test)),
                          accuracy_score(y_test,
                                         fitted_models[0].predict(small_X_test)),
                          accuracy_score(y_test,
                                         fitted_models[1].predict(small_X_test)),
                          accuracy_score(y_test,
                                         fitted_models[2].predict(small_X_test)),
                          accuracy_score(y_test,
                                         pd.DataFrame(model4.predict(small_X_test))\
                                         .idxmax(axis = 1).values )],
                        index = ["Full Logit", "Logistic Regression", "SVC",
                                 "Random Forest", "Neural Network"])

        support.scores_table(X_train, small_X_train)
        
        return out
        
        
        
        
        
        
        
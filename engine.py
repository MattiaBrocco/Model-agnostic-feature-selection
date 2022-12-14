import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

from tensorflow.keras.utils import to_categorical

import multiprocessing
from joblib import delayed
from joblib import Parallel

import support
import evaluation


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
        y = df[target].apply(lambda v: 1 if v == "HONEST"
                             or v == "H" else 0)
        
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
        
        # STEP 0: CREATE A VALIDATION SET
        X_train_no_val, X_val, y_train_no_val, y_val = train_test_split(X_train, y_train, train_size = .8,
                                                                        random_state = 42, shuffle = True)
        
        # STEP 1: Permutation importance with a random forest.
        #         The forest inherits the complexity of a post-pruned tree.
        #         The tree is subjected to post-pruning (with CV)
        
        # 1.1 Pruning        
        pruned_tree = support.D3_pruning(X_train_no_val, y_train_no_val)
        # !!!!!!
        random_forest = GradientBoostingClassifier(random_state = 42,
                                                   ccp_alpha = pruned_tree.best_params_["ccp_alpha"])
        random_forest.fit(X_train_no_val, y_train_no_val)
        
        # 1.3 Feature selection
        
        perm_imp = permutation_importance(random_forest, X_val, y_val, n_repeats = 100,
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
        
        # 2. Train a full and a reduced Logistic Regression and perform Wilks test
        # Rmk: this is done on the whole TRAINING dataset
        logreg_red = LogisticRegression(n_jobs = -1, random_state = 42,
                                        max_iter = 5e3, solver = "saga")
        logreg_red.fit(small_X_train, y_train)
        
        logreg_full = LogisticRegression(n_jobs = -1, random_state = 42,
                                         max_iter = 5e3, solver = "saga")
        logreg_full.fit(X_train, y_train)
        
        # 2.1 Run test
        validation_test = support.likelihood_ratio_test(logreg_red, logreg_full,
                                                        small_X_train,
                                                        X_train, y_train) 
        # 2.2 Consider factor loading
        data_fl = pd.DataFrame(np.c_[small_X_train, y_train]).corr()
        fl = np.abs(data_fl)[len(selected_features)].min()
        
        # validation_test[1]: p-value of likelihood ratio test
        # If p_value > 0, the nested model should be used
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
        
        sel_features = features["Features"] if isinstance(features, dict) else features
        
        
        if isinstance(X_train, pd.DataFrame):
            small_X_train = X_train.iloc[:, sel_features]
        else:
            small_X_train = X_train[:, sel_features]
            
        if isinstance(X_test, pd.DataFrame):
            small_X_test = X_test.iloc[:, sel_features]
        else:
            small_X_test = X_test[:, sel_features]
        
        # Full Logit
        model0 = LogisticRegression(random_state = 42, n_jobs = -1,
                                    max_iter = 5e3, solver = "saga")
        model0.fit(X_train, y_train)
        # Logistic regression
        model1 = LogisticRegression(random_state = 42, n_jobs = -1,
                                    max_iter = 5e3, solver = "saga")
        model1.fit(small_X_train, y_train)
        
        # Support vector machine
        model2 = LinearSVC(random_state = 42, max_iter = 1e4)
        model2.fit(small_X_train, y_train)
        
        # Random forest
        model3 = GradientBoostingClassifier(random_state = 42,
                                            min_samples_leaf = np.max([5,
                                                                       len(X_train)/100]).astype(int))
        model3.fit(small_X_train, y_train)
        
        #fitted_models = Parallel(n_jobs = 8)(delayed(lambda m:
        #                                             m.fit(small_X_train, y_train))(model)
        #                                     for model in [model1, model2, model3])
        
        # Neural network
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        model4 = support.build_MLP(X_train, y_train_cat, features)
        
        n_cores = min(multiprocessing.cpu_count(), 8) # don't use more than 8 cores
        
        
        #importance_rank = Parallel(n_jobs = n_cores)\
        #                    (delayed(lambda m:
        #                             support.sorted_importance_index(m, small_X_train, y_train,
        #                                                             features["Features"]))(model)
        #                     for model in [model1, model2, model3, model4])
        
        importance_rank = [[np.nan]*len(sel_features[:5])] +\
                          [support.sorted_importance_index(m, small_X_train, y_train,
                                                           sel_features)
                           for m in [model1, model2, model3, model4]]
        
        # SCORES
        
        out = np.c_[[accuracy_score(y_test, model0.predict(X_test)),
                          accuracy_score(y_test, model1.predict(small_X_test)),
                          accuracy_score(y_test, model2.predict(small_X_test)),
                          accuracy_score(y_test, model3.predict(small_X_test)),
                          accuracy_score(y_test, pd.DataFrame(model4.predict(small_X_test))\
                                                 .idxmax(axis = 1).values )], importance_rank]
        
        out = pd.DataFrame(out, index = ["Full Logit", "Logistic Regression", "SVC",
                                         "Random Forest", "Neural Network"],
                           columns = ["Accuracy"] + [f"FI {i+1}" for i in
                                                     range(len(importance_rank[0]))])
        
        #out = pd.Series( ,
        #                index = ["Full Logit", "Logistic Regression", "SVC",
        #                         "Random Forest", "Neural Network"])
        
        
        #out = pd.Series( [accuracy_score(y_test, model0.predict(X_test)),
        #                  accuracy_score(y_test,
        #                                 fitted_models[0].predict(small_X_test)),
        #                  accuracy_score(y_test,
        #                                 fitted_models[1].predict(small_X_test)),
        #                  accuracy_score(y_test,
        #                                 fitted_models[2].predict(small_X_test)),
        #                  accuracy_score(y_test,
        #                                 pd.DataFrame(model4.predict(small_X_test))\
        #                                 .idxmax(axis = 1).values )],
        #                index = ["Full Logit", "Logistic Regression", "SVC",
        #                         "Random Forest", "Neural Network"])

        support.scores_table(X_train, small_X_train)
        
        stability = evaluation.stability_metric(out)
        
        return out, stability
        

    def mutual_info(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

        from rpy2 import robjects

        np.savetxt("X.csv", X_data, delimiter=",")
        np.savetxt("y.csv", y_data, delimiter=",", fmt='%s')

        robjects.r("""
        library(praznik)
        X <- read.csv("C:/Users/Asus/Desktop/cognitive_datasets/X.csv",header = FALSE)
        y <- read.csv("C:/Users/Asus/Desktop/cognitive_datasets/y.csv",header = FALSE)
        y <- as.factor(as.vector(y[,1]))
        a <- JMIM(X,y,dim(X)[2])
        sorted_scores <- sort(a$score,decreasing = TRUE)
        sorted_scores_max <- max(sorted_scores)
        sorted_scores_new <- sorted_scores/sorted_scores_max
        best_features = a$selection[names(sorted_scores_new[sorted_scores_new > 0.80])]
        best_features = best_features - 1 
        best_features
        """)

        best_features = [int(x) for x in robjects.globalenv['best_features']]

        return best_features



        
        
        
        
        
        
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepare_data(data_dir, target):
    
    if ".xlsx" in data_dir:
        data_import = pd.read_excel(data_dir)
    elif ".feather" in data_dir:
        data_import = pd.read_feather(data_dir)
    elif ".parquet" in data_dir:
        data_import = pd.read_parquet(data_dir)
    else:
        data_import = pd.read_csv(data_dir)
    
    data_import[target] = data_import[target].apply(lambda v: 1 if v == "HONEST"
                                                    or v == "H" else 0)
    
    return data_import


    
def pca_ds_acc(data):
        
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pca = PCA(0.90, random_state = 42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    n, p = data.shape
    perc = round(p/100*20)
    n_pcs= pca.components_.shape[0]
    
    prop = list((pca.explained_variance_ratio_)/np.sum(pca.explained_variance_ratio_))
    edp = []
    count = 0
    for r in range(n_pcs):
        if count < perc:
            epc = round(perc*prop[r])
            if epc == 0 :
                edp.append(perc-count)
                count = perc
            else:
                edp.append(epc)
                count += epc
        else:
            edp.append(0)

    pca_comp = [abs(pca.components_[x]) for x in range(n_pcs)]
    pca_comp = [list(pca_comp[l]) for l in range(n_pcs)]
    pca_comp_sort = [sorted(pca_comp[k],reverse=True)[:edp[k]] for k in range(n_pcs)]

    most_important_index = []
    for k in range(len(pca_comp_sort)):
        for j in range(len(pca_comp_sort[k])):
            most_important_index.append(pca_comp[k].index(pca_comp_sort[k][j]))

    initial_feature_names = data.columns
    most_important_index = list(set(most_important_index))
    #most_important_names = [initial_feature_names[most_important_index[q]]
    #                        for q in range(len(most_important_index))]

    #sub_col = list(most_important_names)
    #sub_col.append("CONDITION")
    
    return most_important_index

def trts_split(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7,
                                                        random_state = 42, shuffle = True)
    return X_train, X_test, y_train, y_test
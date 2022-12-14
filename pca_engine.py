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


def new_mwu_data(data):
    n, p = data.shape
    X_honest = data[data["CONDITION"] == 1].iloc[:, :-1]
    X_dishonest = data[data["CONDITION"] == 0].iloc[:, :-1]
    indices = []
    for k in range(0, (p-1)):
        U1, s = mannwhitneyu(X_honest.iloc[:, k],
                             X_dishonest.iloc[:, k])
        if s < 0.01:
            indices.append(k)
    indices.append(-1)
    
    return data.iloc[:, indices]


def scree_plot(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pca = PCA(0.70)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(pca.explained_variance_ratio_)
    plt.bar(range(1,len(pca.explained_variance_)+1),
            pca.explained_variance_ )
 
    plt.plot(range(1,len(pca.explained_variance_ )+1),
             np.cumsum(pca.explained_variance_), c='red',
             label='Cumulative Explained Variance')
 
    plt.legend(loc='upper left')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (eignenvalues)')
    plt.title('Scree plot')
 
    plt.show()
    
def pca_ds_acc(data):
        
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pca = PCA(0.70)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_pcs= pca.components_.shape[0]
    pca_comp = [abs(pca.components_[x]) for x in range(n_pcs)]
    pca_comp = [list(pca_comp[l]) for l in range(n_pcs)]
    pca_comp_sort = [sorted(pca_comp[p])[-3:] for p in range(n_pcs)]

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
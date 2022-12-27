import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy_barplot(data, prefix, save = False):
    """
    Parameters
    ----------
    
    data : pd.DataFrame
    save : bool
    prefix : str
    """
    plt.figure(figsize = (18, 7))
    plt.bar(x = data.sort_values("Avg Acc. on selected features",
                                 ascending = False)["Dataset name"],
            height = data.sort_values("Avg Acc. on selected features",
                                      ascending = False)["Avg Acc. on selected features"],
            yerr = data.sort_values("Avg Acc. on selected features",
                                    ascending = False)["Accuracy Std on selected features"],
            alpha = .7, color = "#814DFF")
    plt.xticks(rotation = 35, fontsize = 13)
    plt.yticks(fontsize = 12)
    plt.ylabel("Accuracy\n(mean ± std)", fontsize = 13, rotation = 0, labelpad = 50)
    plt.title("Accuracy on different datasets", fontsize = 18)
    
    if save == True:
        plt.savefig(f".\\images\\{prefix}_accuracy_barplot.jpg", dpi = 300)
    else:
        plt.show()
        
        
def accuracy_std(data, prefix, save = False):
    """
    Parameters
    ----------
    
    data : pd.DataFrame
    save : bool
    prefix : str
    """
    plt.figure(figsize = (18, 6))
    plt.bar(x = data.sort_values("Accuracy Std on selected features",
                                 ascending = False)["Dataset name"],
            height = data.sort_values("Accuracy Std on selected features",
                                      ascending = False)["Accuracy Std on selected features"],
            alpha = .7, color = "#814DFF")
    plt.xticks(rotation = 35, fontsize = 13)
    plt.yticks(fontsize = 12)
    plt.ylabel("Stand. Dev.", fontsize = 13, rotation = 0, labelpad = 30)
    plt.title("Standard Deviation of Accuracies on different datasets", fontsize = 18)
    if save == True:
        plt.savefig(f".\\images\\{prefix}_accuracy_std.jpg", dpi = 300)
    else:
        plt.show()
        
        
def faking_type_comparison(data, prefix, save = False):
    
    data_use = data.copy()
    faking_type = {"BF_df_CTU":"FAKING GOOD",
                   "BF_df_OU":"FAKING GOOD",
                   "BF_df_V":"FAKING GOOD",
                   "DT_df_CC":"FAKING GOOD",
                   "DT_df_JI":"FAKING GOOD",
                   "IADQ_df":"FAKING BAD",
                   "IESR_df":"FAKING BAD",
                   "NAQ_R_df":"FAKING BAD",
                   "PCL5_df":"FAKING BAD",
                   "PHQ9_GAD7_df":"FAKING BAD",
                   "PID5_df":"FAKING BAD",
                   "PRFQ_df":"FAKING GOOD",
                   "PRMQ_df":"FAKING BAD",
                   "RAW_DDDT":"FAKING BAD",
                   "R_NEO_PI":"FAKING GOOD",
                   "sPID-5_df":"FAKING GOOD"}
    
    data_use["Faking Type"] = data_use["Dataset name"].map(faking_type)
    
    figft, axft = plt.subplots(2, 1, figsize = (18, 10))
    
    for n, ftp in enumerate(data_use["Faking Type"].unique()):
        
    
        data_use[data_use["Faking Type"] == ftp].sort_values("Accuracy - Logit all features",
                                                             ascending = False)\
                                                .plot(x = "Dataset name",
                                                      y = ["Accuracy - Logit all features",
                                                           "Accuracy - Logit",
                                                           "Accuracy - SVM", "Accuracy - RF",
                                                           "Accuracy - MLP"],
                                                      kind = "bar", rot = 0,
                                                      ax = axft[n], title = ftp,
                                                      color = ["#2F42FE", "#7A5EEF", "#BD78C9",
                                                               "#E88790", "#FE9031"])
        
        axft[n].axhline(data_use[data_use["Faking Type"] == ftp]\
                        ["Avg Acc. on selected features"].mean(), ls = "--", color = "grey")
    
    if save == True:
        plt.savefig(f".\\images\\{prefix}faking_type_comparison.jpg", dpi = 300)
    else:
        plt.show()
    
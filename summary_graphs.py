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
    plt.ylabel("Stand. Dev.", fontsize = 13, rotation = 0, labelpad = 40)
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
        
        
def approaches_comparison(summary1, summary2, summary3, save = False):
    
    
    summ1 = summary1.copy()
    summ1.columns = [f"{c}_pca" if c != "Dataset name" else c
                     for c in summ1.columns]
    summ2 = summary2.copy()
    summ2.columns = [f"{c}_permimp" if c != "Dataset name" else c
                     for c in summ2.columns]
    summ3 = summary3.copy()
    summ3.columns = [f"{c}_mutinfo" if c != "Dataset name" else c
                     for c in summ3.columns]
    
    summary_merge = summ1.merge(summ2, on = "Dataset name",
                                how = "inner").merge(summ3, on = "Dataset name",
                                                     how = "inner")

    xax_labels = summary_merge["Dataset name"].str.replace("_", " ")\
                 .str.replace("df", "").str.replace("  ", " ")

    fig, ax = plt.subplots(4, 1, figsize = (14, 18))
    summary_merge.rename(columns = {"Avg Acc. on selected features_pca": "PCA",
                                    "Avg Acc. on selected features_mutinfo": "Mutual info.",
                                    "Avg Acc. on selected features_permimp": "Perm. Import."})\
                 .plot(x = "Dataset name", y = ["PCA", "Perm. Import.", "Mutual info."],
                       kind = "bar", rot = 0, ax = ax[0],#, 0],
                       title = "Avg Acc. on selected features",
                       color = ["#243FFF", "#BD78C9", "#FE9031"])

    summary_merge.rename(columns = {"Acc. diff. wrt Full logit_pca": "PCA",
                                    "Acc. diff. wrt Full logit_mutinfo": "Mutual info.",
                                    "Acc. diff. wrt Full logit_permimp": "Perm. Import."})\
                 .sort_values("Perm. Import.", ignore_index = True)\
                 .plot(x = "Dataset name", y = ["PCA", "Perm. Import.", "Mutual info."],
                       kind = "bar", rot = 0, ax = ax[1],#, 0],
                       title = "Acc. diff. wrt Full logit",
                       color = ["#243FFF", "#BD78C9", "#FE9031"]) # PREV. PALETTE: ["#243FFF", "#FF9022", "#FF1F85"]

    summary_merge.rename(columns = {"Accuracy Std on selected features_pca": "PCA",
                                    "Accuracy Std on selected features_mutinfo": "Mutual info.",
                                    "Accuracy Std on selected features_permimp": "Perm. Import."})\
                 .plot(x = "Dataset name", y = ["PCA", "Perm. Import.", "Mutual info."],
                       kind = "bar", rot = 0, ax = ax[2],#, 0],
                       title = "Accuracy Std on selected features",
                       color = ["#243FFF", "#BD78C9", "#FE9031"])

    summary_merge.rename(columns = {"Feat. Top5-Stability_pca": "PCA",
                                    "Feat. Top5-Stability_mutinfo": "Mutual info.",
                                    "Feat. Top5-Stability_permimp": "Perm. Import."})\
                 .plot(x = "Dataset name", y = ["PCA", "Perm. Import.", "Mutual info."],
                       kind = "bar", rot = 0, ax = ax[3],#, 0],
                       title = "Feat. Top5-Stability",
                       color = ["#243FFF", "#BD78C9", "#FE9031"])

    # 5 colors palette: https://coolors.co/2f42fe-7a5eef-bd78c9-e88790-fe9031
    # --> ["#2f42fe", "#7a5eef", "#bd78c9", "#e88790", "#fe9031"]

    ax[0].set_xticklabels(xax_labels, rotation = 42)
    ax[1].set_xticklabels(xax_labels, rotation = 42)
    ax[2].set_xticklabels(xax_labels, rotation = 42)
    ax[3].set_xticklabels(xax_labels, rotation = 42)

    ax[0].set(xlabel = None)
    ax[1].set(xlabel = None)
    ax[2].set(xlabel = None)
    ax[3].set(xlabel = None)

    plt.tight_layout()

    if save == True:
        plt.savefig(f".\\images\\Overall_comparison.jpg", dpi = 300)
    else:
        plt.show()
        
        
def final_table(col, summary1, summary2, summary3):
    
    summary_merge = summary1.rename(columns = dict(zip(summary1.columns,
                                                       [f"{c}_pca" if c != "Dataset name" else c
                                                        for c in summary1.columns])))\
                    .merge(summary2.rename(columns = dict(zip(summary2,
                                                              [f"{c}_permimp" if c != "Dataset name" else c
                                                               for c in summary2.columns]))),
                           on = "Dataset name", how = "inner")\
                    .merge(summary3.rename(columns = dict(zip(summary3.columns,
                                                              [f"{c}_mutinfo" if c != "Dataset name" else c
                                                               for c in summary3.columns]))),
                           on = "Dataset name", how = "inner")
    
    
    out = summary_merge[[f"{col}_pca", f"{col}_permimp",
                         f"{col}_mutinfo"]].describe().loc[["mean", "min", "max"]]
    
    out = out.rename(columns = {f"{col}_pca": "PCA", f"{col}_permimp": "PERM. IMP.",
                                f"{col}_mutinfo": "JMIM"})
    print(col)
    return out.applymap(lambda v:round(v, 4))
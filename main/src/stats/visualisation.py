import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from numpy.typing import NDArray
try:
    from .stat_analyses import cohen_d, cliff_delta
except:
    from stat_analyses import cohen_d, cliff_delta
from scipy.stats import shapiro
import numpy as np
from tqdm import tqdm

#Uncomment to see the time spent on each line using "kernprof -l -v path_to_file"
#@profile
def effect_size_distribution(X_wikipedia : NDArray ,
                             X_universalis : NDArray,
                             columns : list[str],
                             path_figure : str,
                             max_features : int = 10,
                             ) -> tuple[list,list,list,list] :
    
    # Variable that'll store the normality check
    normality = []

    # Variables that'll store the differences
    D, delta = [], []

    for feature in tqdm(range(X_universalis.shape[1])):

        data_wiki, data_universalis = X_wikipedia[:,feature], X_universalis[:, feature]
        #current_column = columns[feature]

        _, pval = shapiro(data_wiki)
        if pval < 0.01:
            _, pval = shapiro(data_universalis)
            if pval < 0.01:
                normality.append(True)
            else:
                normality.append(False)
        else:
            normality.append(False)
        
        D.append(cohen_d(data_wiki, data_universalis))
        delta.append(cliff_delta(data_wiki, data_universalis))

    fig, ax = plt.subplots(nrows=2,
                            figsize=(10,7),
                            sharex=False)
    
    best_columns_D = np.argsort([abs(d) for d,_ in D])[-max_features:][::-1]
    best_columns_cliff = np.argsort([abs(delt) for delt,_ in delta])[-max_features:][::-1]
    
    bars = ax[0].bar(x=np.array(columns)[best_columns_D],
                    height=np.array([d for d,_ in D])[best_columns_D],
                    width=0.4
                    )
    
    # Assign hatch only to non-normal bars
    for bar, is_non_normal in zip(bars, normality):
        if is_non_normal:
            bar.set_hatch('//')  # Use any hatch pattern you prefer

    # Create custom legend handles
    legend_handles = [
        Patch(facecolor='skyblue', edgecolor='black', label='Normal distribution'),
        Patch(facecolor='skyblue', edgecolor='black', hatch='//', label='Non-normal distribution')
        ]
    
    ax[0].legend(handles=legend_handles)
    ax[0].tick_params(axis="x",rotation=45)
    ax[1].bar(x=np.array(columns)[best_columns_cliff],
              height=np.array([delt for delt,_ in delta])[best_columns_cliff],
              width = 0.4)
    ax[1].tick_params(axis="x",rotation=45)

    plt.savefig(path_figure,
                dpi=400,
                bbox_inches="tight")
    
    return (columns, delta, D, normality)


if __name__ == "__main__":
    import pandas as pd 
    import joblib

    df_universalis = pd.read_csv("/media/jbulkatravail/DATA2/JB_HD/Thèse/Coffre Fort Thèse/XP_thèse/Classification/Données/data_universalis.csv",
    index_col=0)

    df_wiki = joblib.load("/media/jbulkatravail/DATA2/JB_HD/Thèse/Coffre Fort Thèse/XP_thèse/Classification/Notebooks/data/df_wiki.joblib")
    df_wiki = pd.DataFrame([df.mean(0) for df in tqdm(df_wiki)])
    df_wiki.columns = [x if not "POS" in x else x.split("POS -")[1] for x in df_wiki.columns]

    df_universalis.fillna(0, inplace=True)
    df_wiki.fillna(0, inplace=True)


    columns = [x for x in df_wiki.columns if x in df_universalis.columns]
    df_wiki = df_wiki.loc[:,columns]
    df_universalis = df_universalis.loc[:,columns]
    effect_size_distribution(df_wiki.to_numpy()[:,:5],
                        df_universalis.to_numpy()[:,:5],
                            columns=columns[:5])
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from numpy.typing import NDArray
from .stat_analyses import cohen_d, cliff_delta
from scipy.stats import shapiro
import numpy as np
from tqdm import tqdm

def effect_size_distribution(X_wikipedia : NDArray ,
                             X_universalis : NDArray,
                             columns : list[str],
                             max_features : int = 10) -> None :
    
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
    
    best_columns_D = np.argsort([d for d,_ in D])[:max_features]
    best_columns_cliff = np.argsort([delt for delt,_ in delta])[:,max_features]
    
    bars = ax[0].bar(x=columns[best_columns_D],
                    y=np.array([d for d,_ in D])[best_columns_D],
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

    ax[1].bar(x=columns[best_columns_cliff],
              y=np.array([delt for delt,_ in delta])[best_columns_cliff])
    

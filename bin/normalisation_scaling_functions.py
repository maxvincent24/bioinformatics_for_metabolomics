import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, MaxAbsScaler

import matplotlib.cm as cm

#import scipy.stats as st
#from sci_analysis import analyze


####################################################################################################
##################################  DISPLAY DATA  ##################################################
####################################################################################################

def print_stats(X):
    
    print('----------X.min().mean()----------\n', X.min().mean(), '\n')
    print('----------X.max().mean()----------\n', X.max().mean(), '\n')
    print('----------X.mean().mean()---------\n', X.mean().mean(), '\n')
    print('----------X.std().mean()----------\n', X.std().mean(), '\n')
    

    
      
def plot_first_distributions_metabolites(X):
    
    fig, axs = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle('Distribution of the first 4 feature values of the peak table', fontsize=20)

    sns.histplot(ax=axs[0], x=X.iloc[:,0], color='lightblue')

    sns.histplot(ax=axs[1], x=X.iloc[:,1], color='lightpink')

    sns.histplot(ax=axs[2], x=X.iloc[:,2], color='lightgreen')

    sns.histplot(ax=axs[3], x=X.iloc[:,3], color='coral')

    
    
def plot_first_distributions_samples(X):
    
    fig, axs = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle('Distribution of the first 4 feature values of the peak table', fontsize=20)

    sns.histplot(ax=axs[0], x=X.iloc[0,:], color='lightblue')

    sns.histplot(ax=axs[1], x=X.iloc[1,:], color='lightpink')

    sns.histplot(ax=axs[2], x=X.iloc[2,:], color='lightgreen')

    sns.histplot(ax=axs[3], x=X.iloc[3,:], color='coral')

    
    
    
def boxplot_first_distributions_metabolites(X, list_features=None, n=10):
    
    if list_features == None:
        # list_features = X.columns
        data = X.iloc[:, :n]
        title = f'Boxplots of {n} first features distributions'
    else:
        data = X.loc[:, list_features]
        title = f'Boxplots of selected features distributions'
    
    fig, ax = plt.subplots(figsize=(15, 10))
    bp = ax.boxplot(data, notch=True, patch_artist=True)
    ax.set_xticklabels(data.columns, rotation=70)
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='Features',
        ylabel='Values',
    )
    
    cmap = cm.ScalarMappable(cmap='rainbow')
    data_mean = data.mean()
    for patch, color in zip(bp['boxes'], cmap.to_rgba(data_mean)):
        patch.set_facecolor(color)
    
    plt.show()
    

        

def boxplot_first_distributions_metabolites_before_after(X_init, X_norm, list_features=None, n=10):
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Before / After')
    
    if list_features == None:
        # list_features = X.columns
        data_init = X_init.iloc[:, :n]
        data_norm = X_norm.iloc[:, :n]
        title = f'Boxplots of {n} first features distributions'
    else:
        data_init = X_init.loc[:, list_features]
        data_norm = X_norm.loc[:, list_features]
        title = f'Boxplots of selected features distributions'
    
    #data_init = X_init.iloc[:, :n]
    #data_norm = X_norm.iloc[:, :n]
    
    bp_init = ax1.boxplot(data_init, notch=True, patch_artist=True)
    bp_norm = ax2.boxplot(data_norm, notch=True, patch_artist=True)
    
    ax1.set_xticklabels(data_init.columns[:n], rotation=70)
    ax2.set_xticklabels(data_norm.columns[:n], rotation=70)
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='Features',
        ylabel='Values',
    )
    
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax2.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='Features',
        ylabel='Values',
    )
    
    cmap = cm.ScalarMappable(cmap='rainbow')
    
    data_init_mean = data_init.mean()
    data_norm_mean = data_norm.mean()
    
    for patch, color in zip(bp_init['boxes'], cmap.to_rgba(data_init_mean)):
        patch.set_facecolor(color)
    
    for patch, color in zip(bp_norm['boxes'], cmap.to_rgba(data_norm_mean)):
        patch.set_facecolor(color)
    
    
    plt.show()        





    
    
####################################################################################################
#############################  TRANSFORMATION METHODS  #############################################
####################################################################################################

def do_log_e(arg):
    return np.log(arg)

def do_log_2(arg):
    return np.log2(arg)

def do_log_10(arg):
    return np.log10(arg)

def do_sqrt(arg):
    return np.sqrt(arg)

def do_cbrt(arg):
    return np.cbrt(arg)

    
    
    
####################################################################################################
#####################  METABOLITE-BASED NORMALISATION METHODS  #####################################
####################################################################################################  

def do_autoscaling(arg):
    
    #col_means = np.mean(arg, axis=0)
    #col_means = np.array(col_means).reshape((1, col_means.shape[0]))
    #col_std = np.std(arg, axis=0)
    #col_std = np.array(col_std).reshape((1, col_std.shape[0]))
    #return (arg - col_means) / col_std
    
    return pd.DataFrame(StandardScaler().fit_transform(arg), columns=arg.columns)
    

    
def do_range_scaling(arg):
    
    col_min = np.min(arg, axis=0)
    col_min = np.array(col_min).reshape((1, col_min.shape[0]))
    col_max = np.max(arg, axis=0)
    col_max = np.array(col_max).reshape((1, col_max.shape[0]))
    
    return (arg - col_min) / (col_max - col_min)
    
    
    
def do_minmax_scaling(arg):
    
    return pd.DataFrame(MinMaxScaler().fit_transform(arg), columns=arg.columns)



def do_pareto_scaling(arg):
    
    col_means = np.mean(arg, axis=0)
    col_means = np.array(col_means).reshape((1, col_means.shape[0]))
    col_std = np.std(arg, axis=0)
    col_std = np.array(col_std).reshape((1, col_std.shape[0]))
    
    return (arg - col_means) / np.sqrt(col_std)



def do_vast_scaling(arg):

    col_means = np.mean(arg, axis=0)
    col_means = np.array(col_means).reshape((1, col_means.shape[0]))
    col_std = np.std(arg, axis=0)
    col_std = np.array(col_std).reshape((1, col_std.shape[0]))
    X_autoscaled = (arg - col_means) / col_std
    
    return X_autoscaled * (col_means / col_std)
    
    

def do_level_scaling(arg):
    
    col_means = np.mean(arg, axis=0)
    col_means = np.array(col_means).reshape((1, col_means.shape[0]))
    
    return (arg - col_means) / col_means
    

    
def do_max_abs_scaling(arg):
    
    #col_max = np.max(arg, axis=0)
    #col_max = np.array(col_max).reshape((1, col_max.shape[0]))
    
    #return arg / col_max

    return pd.DataFrame(MaxAbsScaler().fit_transform(arg), columns=arg.columns)
    


def do_robust_scaling(arg):
    
    return pd.DataFrame(RobustScaler().fit_transform(arg), columns=arg.columns)
    





    
####################################################################################################
#######################  SAMPLE-BASED NORMALISATION METHODS  #######################################
####################################################################################################    
    
def do_mean_normalisation(arg):
    
    row_means = np.mean(arg, axis=1)
    row_means = np.array(row_means).reshape((row_means.shape[0], 1))
    return arg - row_means

def do_median_normalisation(arg):
    
    row_medians = np.median(arg, axis=1)
    row_medians = np.array(row_medians).reshape((row_medians.shape[0], 1))
    return arg - row_medians


def do_l1_normalisation(arg):

    return pd.DataFrame(Normalizer(norm='l1').fit_transform(arg), columns=arg.columns)


def do_l2_normalisation(arg):
    
    return pd.DataFrame(Normalizer(norm='l2').fit_transform(arg), columns=arg.columns)
    
    
    

    
    
    
    
    
    
####################################################################################################
####################################  PIPELINE  ####################################################
####################################################################################################  

def normPeakTable(X, method, based):
    
    dict_methods = {
        'loge': do_log_e,
        'log2': do_log_2,
        'log10': do_log_10,
        'sqrt': do_sqrt,
        'cbrt': do_cbrt,
        'range_scaling': do_range_scaling,
        'minmax_scaling': do_minmax_scaling,
        'max_abs_scaling': do_max_abs_scaling,
        'level_scaling': do_level_scaling,
        'robust_scaling': do_robust_scaling,
        'autoscaling': do_autoscaling,        
        'pareto_scaling': do_pareto_scaling,
        'vast_scaling': do_vast_scaling,
        'mean_normalisation': do_mean_normalisation,
        'median_normalisation': do_median_normalisation,
        'l1_normalisation': do_l1_normalisation,
        'l2_normalisation': do_l2_normalisation
    }
    
    X_norm = dict_methods[method](X)
    
    return X_norm






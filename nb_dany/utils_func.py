from matplotlib.ticker import MaxNLocator # needed for integer only on axis
from matplotlib.lines import Line2D # for creating the custom legend
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid') # set style
#import libraries
import numpy as np
import pandas as pd
#Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#scipy
from scipy.stats.mstats import winsorize
#test-training learning curve for each kNN and check low/high variance model. 
from sklearn.model_selection import learning_curve

# #feature scaling
def scaler_transform(scaler_type, X, exclude_vars = ['latitude', 'longitude', 'id','NG_Brooklyn','NG_Manhattan','NG_Queens','NG_Bronx','NG_Staten Island','room_type_Private room','room_type_Shared room']):
    '''
    Transform the NON-object type data to the selected scaler
    '''
    X_copy = X.copy(deep = True)
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    X_notexcluded= X_copy.loc[:, ~X_copy.columns.isin(exclude_vars)]
    X_withoutobj = X_notexcluded.select_dtypes(exclude=['object'])
    X_withoutobj = X_withoutobj.add_suffix(f'_{scaler_type}')
    # Fit transform the scaler if there are objects in dataset
    if X_withoutobj.shape[1] > 0:
        X_withoutobj = pd.DataFrame(scaler.fit_transform(X_withoutobj[X_withoutobj.columns]),
                                        index=X_withoutobj.index,
                                        columns=X_withoutobj.columns)
        # Concatenate the rest of the data
        X_withexcl = pd.concat([X_withoutobj, X.loc[:, X.columns.isin(exclude_vars)]], axis=1)
        others = [x for x in X.select_dtypes('object') if x not in X_withexcl.columns]
        X_final = pd.concat([X_withexcl,X[others]], axis=1)
    return X_final, scaler


def inverse_get_dummies(one_hot_columns,rest_columns,rename):
    """ Reverse function to get_dummies.

    Args:
        one_hot_columns ([list]): list of categorical variables that are dummies.
        df_target_column ([list]): list of variables to not transform
        rename ([str]): column name with categorical values. 

    Returns:
        pd.DataFrame: returns the dataframe with the categorical variables in the original format.
    """    
    data = pd.concat([rest_columns,pd.DataFrame(one_hot_columns[(one_hot_columns==1)].stack().reset_index().set_index('level_0')).drop(0,axis=1)],axis=1)
    data.rename(columns = {'level_1':rename},inplace = True)
    return data

def decision_boundary_plot(classifier, X: pd.DataFrame, y: pd.DataFrame,  classes: list, 
                             h: float = 0.1, prob_dot_scale: int = 20, prob_dot_scale_power: int = 2,
                             true_dot_size: int = 30, pad: float = 1.0,
                             prob_values: list = [0.3, 0.6,  1.0]) -> None:
    """THis function creates a decision boundary between specified classes 
    Args:
        classifier (_type_): the classifier we want to visualize the decision boundary for.
        X (pd.DataFrame): data to plot
        y (pd.DataFrame): target to plot
        classes (list): target labels
        h (float, optional): mesh stepsize. Defaults to 0.1.
        prob_dot_scale (int, optional): modifier to scale the probability dots. Defaults to 20.
        prob_dot_scale_power (int, optional): exponential used to increase or decrease size of prob dots. Defaults to 2.
        true_dot_size (int, optional): size of the true labels. Defaults to 30.
        pad (float, optional): setting bounds of plot. Defaults to 1.0.
        prob_values (list, optional): _description_. Defaults to [0.4, 0.6, 0.8, 1.0].
        
    """    
    # creating meshgrid 
    x0_min, x0_max = np.round(X.iloc[:,0].min())-pad, np.round(X.iloc[:,0].max()+pad)
    x1_min, x1_max = np.round(X.iloc[:,1].min())-pad, np.round(X.iloc[:,1].max()+pad)
    x0_axis_range = np.arange(x0_min,x0_max, h)
    x1_axis_range = np.arange(x1_min,x1_max, h)
    xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)
    #reshaping x into 2D array
    xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))
    # prediction
    yy_hat = classifier.predict(xx) 
    # probability of each observation belonging to each class
    yy_prob = classifier.predict_proba(xx)         
    # size of the dot
    yy_size = np.max(yy_prob, axis=1) 
    #create figure plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,10), dpi=150)
    # establish colors and colormap
    colormap = np.array(['tab:blue', 'tab:purple', 'tab:olive'])
    #plotting all dots
    ax.scatter(xx[:,0], xx[:,1], c=colormap[yy_hat],  alpha=0.4, 
           s=prob_dot_scale*yy_size**prob_dot_scale_power, linewidths=0,)
    ax.contour(x0_axis_range, x1_axis_range, 
           np.reshape(yy_hat,(xx0.shape[0],-1)), 
           levels=3, linewidths=1, 
           colors=colormap)
    ax.scatter(X.iloc[:,0], X.iloc[:,1], c=colormap[y], s=true_dot_size, zorder=3, linewidths=0.7, edgecolor='k')
    #axis legends
    ax.set_ylabel(f"{X.iloc[:,1].name}")
    ax.set_xlabel(f"{X.iloc[:,0].name}")
    
    legend_class = []
    for rt, color in zip(classes, colormap):
        legend_class.append(Line2D([0], [0], marker='o', label=rt,ls='None',
                                markerfacecolor=color, markersize=np.sqrt(true_dot_size), 
                                markeredgecolor='k', markeredgewidth=0.7))
        
    legend_prob = []
    for prob in prob_values:
        legend_prob.append(Line2D([0], [0], marker='o', label=prob, ls='None', alpha=0.8,
                                markerfacecolor='grey', 
                                markersize=np.sqrt(prob_dot_scale*prob**prob_dot_scale_power), 
                                markeredgecolor='k', markeredgewidth=0))
    legend1 = ax.legend(handles=legend_class, loc='center', 
                bbox_to_anchor=(1.05, 0.35),
                frameon=True, title='label')

    legend2 = ax.legend(handles=legend_prob, loc='center', 
                bbox_to_anchor=(1.05, 0.65),
                frameon=True, title='probability', )
    
        
    ax.add_artist(legend1)
    ax.grid(False)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(ax.get_xticks()[1:-1])
    ax.set_yticks(np.arange(x1_min,x1_max, 1)[1:])
    ax.set_aspect(1)
    plt.tight_layout()
    plt.show()



def plot_learning_curve( estimator, title, X, y, axes = None, ylim=None, cv=None, n_jobs=None,train_sizes=np.linspace(0.1, 1.0, 5)):
    #initial configurations for the plot
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    #learning curve creation and statistics 
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator,X,y,cv = cv, n_jobs= n_jobs, train_sizes= train_sizes, return_times = True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plotting learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot the samples vs fit times.
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, "o-")
    # axes[1].fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    # )
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    # )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,10),fontsize = 14, model_name = 'Model'):
    df_cm = pd.DataFrame(confusion_matrix, index = class_names, columns = class_names)
    try: 
      plt.figure()
      heatmap = sns.heatmap(df_cm, annot= True, cmap = "Blues",fmt = '0.0f')
    except ValueError: 
        raise ValueError('Confusion Matrix values must be integers')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    return plt


import seaborn as sns
def hist_box_plot(df, col):
	sns.set(style="darkgrid")
	f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
	sns.boxplot(df[col], ax=ax_box)
	sns.histplot(data=df[col], ax=ax_hist)
	ax_box.set(xlabel='')
	return plt.show()
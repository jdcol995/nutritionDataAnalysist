from matplotlib import pyplot as plt
import seaborn as sns

def plot_distributions(dist_1, dist_2, method, path):
    """
    function to plot the kernel density estimation
    between the synthetic data distribution and the
    original distribution, it takes only two
    parameters as input and has no return
    """
    fig = plt.figure(figsize=(20, 20))
    
    for i, column in enumerate(dist_1.columns, 1):
        ax = plt.subplot(2, 5, i)
        sns.kdeplot(dist_1[column], fill=True, label="Real", color='r', ax=ax)
        sns.kdeplot(dist_2[column], fill=True, label="Synthetic", color='b', ax=ax)
        ax.set_xlabel(f'{column}')
        ax.set_ylabel('')
        ax.set_title(f"Distribution for {column} with " + method, fontsize=8)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15)
    plt.savefig(path + 'Distributions_'+method) 



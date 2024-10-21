import matplotlib.pyplot as plt
import seaborn as sns

def plot_label_distribution(data, X_train, X_test):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Original dataset distribution
    sns.countplot(x=data['label'], ax=ax[0])
    ax[0].set_title('Original Dataset Label Distribution')

    # Training dataset distribution
    train_labels = data[data['path'].isin(X_train)]['label']
    sns.countplot(x=train_labels, ax=ax[1])
    ax[1].set_title('Training Set Label Distribution')

    plt.tight_layout()
    plt.show()


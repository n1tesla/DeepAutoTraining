import seaborn as sns
from matplotlib import pyplot as plt
def heatmap(df):
    sns.heatmap(df.corr(),annot=True,cmap="RdY1Gn",linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,20)
    return plt.show()


def save_fig(history, model_path, plots_path, i, observation_name):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    lr=history.history['lr']

    epochs = range(len(loss))  # Get number of epochs
    plt.rcParams["figure.figsize"] = (16, 8)
    fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3)
    ax1.plot(epochs, loss, 'b', label='loss')
    ax1.plot(epochs, val_loss, 'g', label='val_loss')
    ax1.set_title("Loss")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid()
    ax2.plot(epochs, acc, label='train_acc')
    ax2.plot(epochs, val_acc, label='val_acc')
    ax2.set_title("Train vs Val Accuracy")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()
    ax3.plot(epochs,lr,label='learning_rate')
    ax3.set_title("learning rate")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("learning_rate")
    ax3.legend()
    ax3.grid()

    plt.savefig(model_path + f"/LossAcc_{observation_name}_{i}.png", format='png')
    plt.savefig(plots_path + f"/LossAcc_{observation_name}_{i}.png", format='png')
    plt.close()
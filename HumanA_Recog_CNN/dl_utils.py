import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

def plot_metrics(metrics):
    
    epochs = [e+1 for e in metrics.epoch]
    #epochs = array(metrics.epoch) + 1
    train_loss = metrics.history['loss']
    val_loss = metrics.history['val_loss']
    train_acc = metrics.history['acc']
    val_acc = metrics.history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    ax1.plot(epochs, train_loss, color='red', label='train_loss')
    ax1.plot(epochs, val_loss, color='green', label='val_loss')
    #ax1.set_xticks(epochs)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('cross_entropy(multiclass_logloss)')
    ax1.set_title('LOSS')
    ax1.grid()
    ax1.legend()

    ax2.plot(epochs, train_acc, color='red', label='train_acc')
    ax2.plot(epochs, val_acc, color='green', label='val_acc')
    #ax2.set_xticks(epochs)
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('ACCURACY')
    ax2.grid()
    ax2.legend()
    
    plt.show()
    


def plot_confusion_matrix(Ytrue, Ypred, labels=None):
    
    C = confusion_matrix(Ytrue, Ypred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    heatmap(C, annot=True, cmap='Spectral_r', fmt='d',
            xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=15)
    ax1.set_xlabel('PREDICTED_LABEL', fontsize=12)
    ax1.set_ylabel('TRUE_LABEL', fontsize=12)
    
    
    heatmap(C/C.sum(1), annot=True, cmap='Spectral_r', fmt='.3f',
            xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax2)
    ax2.set_title('Normalized Confusion Matrix', fontsize=15)
    ax2.set_xlabel('PREDICTED_LABEL', fontsize=12)
    ax2.set_ylabel('TRUE_LABEL',fontsize=12)

    fig.tight_layout()
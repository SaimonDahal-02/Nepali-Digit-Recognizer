import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from loader.data_loader import dataloader
from ml.cnn.model import ConvNeuralNetwork

def test():
    model = ConvNeuralNetwork()
    model.load_state_dict(torch.load('models/adam_model.pth'))
    _, _, test_loader = dataloader()
    
    y_true = []
    y_pred = []

    # since we are not training, we don't need to calculate the gradients for outputs
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes, filename=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Get true and predicted labels
y_true, y_pred = test()

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

classes = [str(i) for i in range(10)]

plot_confusion_matrix(y_true, y_pred, classes, filename="plots/adam_confusion_matrix.png")

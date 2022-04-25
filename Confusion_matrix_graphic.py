import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10], :]
    cm = cm[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
    print('Confusion matrix')
    classes = ('bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal')
    plt.figure(figsize=(11,11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.3f}".format(cm[i, j]), horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, fig, ax, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10], :]
    cm = cm[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
    print('Confusion matrix')
    classes = ('bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal')
    # plt.figure(1, figsize=(11,11))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.title(title)
    ax.colorbar()
    tick_marks = np.arange(len(classes))
    ax.xticks(tick_marks, classes, rotation=45)
    ax.yticks(tick_marks, classes)

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, "{:.3f}".format(cm[i, j]), horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")

    ax.tight_layout()
    ax.ylabel('True label')
    ax.xlabel('Predicted label')

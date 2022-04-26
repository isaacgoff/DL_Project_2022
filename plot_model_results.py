import matplotlib.pyplot as plt
import torch


def plot_model_results(epoch_results, model_name):
    if model_name == 'unspecified':
        model_name = f'unspecified-{torch.randn()}'

    # Plot training and validation loss by epoch
    # Create lists for plotting
    epochs, tng_losses, val_losses, tng_acc, val_acc = [], [], [], [], []
    for epoch in epoch_results:
        epochs.append(epoch["epoch"])
        tng_losses.append(epoch["tng_loss"])
        val_losses.append(epoch["val_loss"])
        tng_acc.append(epoch["tng_acc"])
        val_acc.append(epoch["val_acc"])

    # Code to plot loss values by epoch
    plt.plot(epochs, tng_losses, label=f'Training Loss')
    plt.plot(epochs, val_losses, label=f'Validation Loss')
    plt.plot(epochs, tng_acc, label=f'Training Accuracy')
    plt.plot(epochs, val_acc, label=f'Validation Accuracy')
    plt.title(f'Model Results by Epoch')
    plt.xlabel(f'Epoch')
    plt.ylabel(f'Loss and Accuracy')
    plt.legend()
    plt.axis([0, len(epochs), 0, 3])
    # plt.show()
    plt.savefig(f'/content/drive/MyDrive/DL_data/Results/{model_name}-plot.png', dpi=150, bbox_inches='tight', facecolor='gray')
    plt.clf()

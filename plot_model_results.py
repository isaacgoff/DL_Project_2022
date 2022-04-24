import matplotlib.pyplot as plt


def plot_model_results(epoch_results):

    # Plot training and validation loss by epoch
    # Create lists for plotting
    epochs, tng_losses, val_losses, tng_acc, val_acc = [], [], [], [], []
    for epoch in epoch_results:
        epochs.append(epoch_results["epoch"])
        tng_losses.append(epoch_results["tng_loss"])
        val_losses.append(epoch_results["val_loss"])
        tng_acc.append(epoch_results["tng_acc"])
        val_acc.append(epoch_results["val_acc"])

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
    plt.show()
    # plt.savefig(f'{filename}-plot_loss.png', dpi=150, bbox_inches='tight', facecolor='gray')
    plt.clf()


import matplotlib.pyplot as plt


def plot_model_results(epoch_results):

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
    plt.figure(1)
    plt.figure(1).plot(epochs, tng_losses, label=f'Training Loss')
    plt.figure(1).plot(epochs, val_losses, label=f'Validation Loss')
    plt.figure(1).plot(epochs, tng_acc, label=f'Training Accuracy')
    plt.figure(1).plot(epochs, val_acc, label=f'Validation Accuracy')
    plt.figure(1).title(f'Model Results by Epoch')
    plt.figure(1).xlabel(f'Epoch')
    plt.figure(1).ylabel(f'Loss and Accuracy')
    plt.figure(1).legend()
    plt.figure(1).axis([0, len(epochs), 0, 3])
    plt.figure(1).show()
    # plt.savefig(f'/content/drive/MyDrive/DL_data/plot-results.png', dpi=150, bbox_inches='tight', facecolor='gray')
    # plt.clf()

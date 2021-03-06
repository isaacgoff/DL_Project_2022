'''Module that instantiates a deep learning model using the Models class and trains the model for the 
specified number of epochs. Model weights are saved after the epoch with the highest validation accuracy.'''

import argparse
import torch
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from copy import deepcopy
from create_dataset import create_dataset
from Models import Models
from plot_model_results import plot_model_results


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--train_folder', type=str, default='small_audio/')
    parser.add_argument('--val_folder', type=str, default='small_audio/')
    parser.add_argument('--model', type=str, default='Basic_4_Layer_CNN')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--status_interval', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='unspecified')
    parser.add_argument('--save_model', type=str, default='False')
    parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--num_mels', type=int, default=64)
    parser.add_argument('--num_fft', type=int, default=2048)
    parser.add_argument('--hop_len', type=int, default=1000)
    parser.add_argument('--sources', type=str, default='aes')
    args = parser.parse_args()

    if args.save_model.lower() == 'true':
        save_trained_model = True
    else:
        save_trained_model = False

    if args.sources.lower() == 'a':
        sources = [0]
    elif args.sources.lower() == 'ae':
        sources = [0,1]
    elif args.sources.lower() == 'aes':
        sources = [0,1,2]
    elif args.sources.lower() == 'as':
        sources = [0,2]
    elif args.sources.lower() == 'e':
        sources = [1]
    elif args.sources.lower() == 's':
        sources = [2]

    drive_path = '/content/drive/MyDrive/DL_data/'
    json_path_tng = f'{drive_path}nsynth-train/examples.json'
    json_path_val = f'{drive_path}nsynth-valid/examples.json'
    audio_input_path_tng = f'/content/{args.train_folder}'
    audio_input_path_val = f'{drive_path}nsynth-valid/{args.val_folder}'

    # Select GPU for runtime if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print('No GPU selected')
    else:
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))

    start = datetime.now()
    # Create datasets
    tng_dataset = create_dataset(audio_input_path_tng, json_path_tng, args.num_mels, args.hop_len, sources)
    val_dataset = create_dataset(audio_input_path_val, json_path_val, args.num_mels, args.hop_len, sources)

    # Create Data Loaders

    tng_dataloader = DataLoader(tng_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print(f'\nDatasets created in {datetime.now()-start}')

    # Load model
    net = Models(args.model).choose_model().to(device)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # Choose optimizer based on input
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialise loss function for training
    loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_factor)

    epoch = 0
    epoch_results =[]
    for i in range(args.num_epochs):
        epoch_tng_loss = 0
        epoch_tng_acc = 0
        epoch_val_loss = 0
        epoch_val_acc = 0
        epoch_result = {'epoch': epoch}

        # Training Loop
        net.train()
        n = 0
        # print(f'\n*** TRAINING LOOP ***\n')
        for (img_batch, label_batch) in tng_dataloader:
            # print(img_batch.shape)

            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            # print(f'img_batch:\n{img_batch}\nlabel_batch ({label_batch.shape}):\n{label_batch}')

            img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
            # print(f'img_batch shape: {img_batch.shape}')
            predicted_labels = net(img_batch)
            # print(f'predicted_labels ({predicted_labels.shape}): {predicted_labels}')

            tng_loss = loss(predicted_labels, label_batch)
            tng_loss.backward()
            optimizer.step()
            epoch_tng_loss += float(tng_loss.detach().item())
            with torch.no_grad():
                epoch_tng_acc += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
            n += len(label_batch)

        # print(f'\nn = {n}')
        epoch_tng_loss /= len(tng_dataloader)
        epoch_tng_acc /= n
        epoch_result['tng_loss'] = "{:.4f}".format(epoch_tng_loss)
        epoch_result['tng_acc'] = "{:.4f}".format(epoch_tng_acc)

        # Validation Loop
        # print(f'\n*** VALIDATION LOOP ***\n')
        with torch.no_grad():
            net.eval()
            n = 0
            # confusion_matrix = torch.zeros(11, 11)
            for (img_batch, label_batch) in val_dataloader:
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
                # print(f'img_batch:\n{img_batch}\nlabel_batch:\n{label_batch}')

                img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
                predicted_labels = net(img_batch)
                # print(f'predicted_labels: {predicted_labels}')

                val_loss = loss(predicted_labels, label_batch)
                epoch_val_loss += float(val_loss.item())
                epoch_val_acc += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
                n += len(label_batch)

                # calculate confusion matrix elements
                # for j in range(len(label_batch)):
                #     confusion_matrix[torch.argmax(label_batch[j, :])][torch.argmax(predicted_labels[j, :])] += 1

            # print(f'\nn = {n}')
            epoch_val_loss /= len(val_dataloader)
            epoch_val_acc /= n
            epoch_result['val_loss'] = "{:.4f}".format(epoch_val_loss)
            epoch_result['val_acc'] = "{:.4f}".format(epoch_val_acc)
            # label_counts = torch.sum(confusion_matrix, dim=1).reshape(len(confusion_matrix), 1)
            # confusion_matrix /= label_counts

        epoch_results.append(epoch_result)
        if epoch % args.status_interval == 0:
            print(f'epoch {epoch} completed: Training Loss = {"{:.4f}".format(epoch_tng_loss)} //'
                  f' Training Acc = {"{:.4f}".format(epoch_tng_acc)} // '
                  f'Validation Acc = {"{:.4f}".format(epoch_val_acc)}')

        # Establish training cutoff criteria
        if epoch == 0:
            max_val_acc = epoch_val_acc
            best_model_state = deepcopy(net.state_dict())
            # best_confusion_matrix = confusion_matrix
            best_epoch = epoch
        elif epoch_val_acc > max_val_acc:
            # print(f'new minimum loss achieved at epoch {epoch}', file=output_file)
            max_val_acc = epoch_val_acc
            best_model_state = deepcopy(net.state_dict())  # Save state of model with minimum validation loss
            # best_confusion_matrix = confusion_matrix
            best_epoch = epoch

        epoch += 1

    # Call function to generate performance data
    print(f'\nBest Epoch: {best_epoch}')
    print(f'Training Loss = {epoch_results[best_epoch]["tng_loss"]} // '
          f'Training Acc = {epoch_results[best_epoch]["tng_acc"]} '
          f'// Validation Acc = {epoch_results[best_epoch]["val_acc"]}')
    plot_model_results(epoch_results, args.model_name)

    # Save the best model state for future use
    if save_trained_model:
        torch.save(best_model_state, f'{drive_path}nsynth-models/{args.model_name}')

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()

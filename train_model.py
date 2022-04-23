import argparse
import torch
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from create_dataset import create_dataset
from Models import Models
from BasicCNN import BasicCNN


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--train_folder', type=str, default='small_audio/')
    parser.add_argument('--val_folder', type=str, default='small_audio/')
    parser.add_argument('--model', type=str, default='Basic_4_Layer_CNN')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--status_interval', type=int, default=1)
    args = parser.parse_args()

    drive_path = '/content/drive/MyDrive/DL_data/'
    json_path_tng = f'{drive_path}nsynth-train/examples.json'
    json_path_val = f'{drive_path}nsynth-valid/examples.json'
    audio_input_path_tng = f'{drive_path}nsynth-train/{args.train_folder}'
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
    tng_dataset = create_dataset(audio_input_path_tng, json_path_tng)
    val_dataset = create_dataset(audio_input_path_val, json_path_val)

    # Create Data Loaders
    tng_dataloader = DataLoader(tng_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    # net = Models('Basic_4_Layer_CNN')
    net = BasicCNN().to(device)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # X = torch.rand(size=(1, 1, 128, 128))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    epoch = 0
    for epoch in range(args.num_epochs):
        epoch_tng_loss = 0
        epoch_tng_score = 0
        epoch_val_score = 0

        # Training Loop
        net.train()
        n = 0
        print(f'\n*** TRAINING LOOP ***\n')
        for (img_batch, label_batch) in tng_dataloader:
            print(img_batch.shape)

            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            print(f'img_batch:\n{img_batch}\nlabel_batch ({label_batch.shape}):\n{label_batch}')

            img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
            # print(f'img_batch shape: {img_batch.shape}')
            predicted_labels = net(img_batch)
            print(f'predicted_labels ({predicted_labels.shape}): {predicted_labels}')

            tng_loss = loss(predicted_labels, label_batch)
            tng_loss.backward()
            optimizer.step()
            epoch_tng_loss += float(tng_loss.detach().item())
            with torch.no_grad():
                epoch_tng_score += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
            n += len(label_batch)

        print(f'\nn = {n}')
        epoch_tng_loss /= len(tng_dataloader)
        epoch_tng_score /= n

        # Validation Loop
        print(f'\n*** VALIDATION LOOP ***\n')
        with torch.no_grad():
            net.eval()
            n = 0
            for (img_batch, label_batch) in val_dataloader:
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
                print(f'img_batch:\n{img_batch}\nlabel_batch:\n{label_batch}')

                img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
                predicted_labels = net(img_batch)
                print(f'predicted_labels: {predicted_labels}')

                epoch_val_score += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
                n += len(label_batch)

            print(f'\nn = {n}')
            epoch_val_score /= n

        if epoch % args.status_interval == 0:
            print(f'\nepoch {epoch} completed: Training Loss = {epoch_tng_loss} //'
                  f' Training Score = {epoch_tng_score} // Validation Score = {epoch_val_score}')

        epoch += 1

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()

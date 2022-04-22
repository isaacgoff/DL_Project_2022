import argparse
import torch
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from create_dataset import create_dataset
from BasicCNN import BasicCNN


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--train_folder', type=str, default='small_audio/')
    parser.add_argument('--val_folder', type=str, default='small_audio/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--num_epochs', type=int, default=50)
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

    # for batch in tng_dataloader:
    #     print(f'label ({label.shape}):{label}\nimg ({batch["img"].shape}):\n{batch["img"]}')

    # Load model
    net = BasicCNN().to(device)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    epoch = 0
    for epoch in range(args.num_epochs):
        epoch_tng_loss = 0
        epoch_tng_score = 0
        epoch_val_score = 0

        # Training Loop
        net.train()
        n = 0
        for (img_batch, label_batch) in tng_dataloader:
            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            predicted_labels = net(img_batch)

            tng_loss = loss(predicted_labels, label_batch)
            tng_loss.backward()
            optimizer.step()
            epoch_tng_loss += float(tng_loss.detach().item())
            epoch_tng_score += (predicted_labels.argmax(axis=1) == label_batch).sum().detach.item()
            n += len(label_batch)

        epoch_tng_loss /= len(tng_dataloader)
        epoch_tng_score /= n

        # Validation Loop
        with torch.no_grad():
            net.eval()
            n = 0
            for (img_batch, label_batch) in val_dataloader:
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)

                predicted_labels = net(img_batch)

                epoch_val_score += (predicted_labels.argmax(axis=1) == label_batch).sum().item()
                n += len(label_batch)
            epoch_val_score /= n

        if epoch % args.status_interval == 0:
            print(f'\nepoch {epoch} completed: Training Loss = {epoch_tng_loss} //'
                  f' Training Score = {epoch_tng_score} // Validation Score = {epoch_val_score}')

        epoch += 1

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()

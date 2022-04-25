import argparse
import torch
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from create_dataset import create_dataset
from Models import Models

from Confusion_matrix_graphic import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Test the individual instrument identification model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--test_folder', type=str, default='audio/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', type=str,  default='billy_joel_cnn')
    parser.add_argument('--model_type', type=str, default='Basic_4_Layer_CNN')
    args = parser.parse_args()

    # Dataset and model paths
    drive_path = '/content/drive/MyDrive/DL_data/'
    json_path_test = f'{drive_path}nsynth-test/examples.json'
    audio_input_path_test = f'{drive_path}nsynth-test/{args.test_folder}'
    model_path = f'{drive_path}nsynth-models/{args.model_name}'

    # Select GPU for runtime if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print('No GPU selected')
    else:
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))

    # Add code to store and deal with output files

    start = datetime.now()

    # Create dataset
    test_dataset = create_dataset(audio_input_path_test, json_path_test)

    # Create Data Loader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f'\nDatasets created in {datetime.now()-start}')

    # Create instance of model and load saved weights
    model = Models(args.model_type).choose_model().to(device)
    model.load_state_dict(torch.load(f'{model_path}', map_location=device))

    # Inference loop
    print("Beginning inference loop\n.....")
    test_score = 0
    with torch.no_grad():
        model.eval()
        n = 0

        confusion_matrix = torch.zeros(11,11)

        for (img_batch, label_batch) in test_dataloader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            # print(f'img_batch:\n{img_batch}\nlabel_batch:\n{label_batch}')

            img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
            predicted_labels = model(img_batch)
            # print(f'predicted_labels: {predicted_labels}')

            test_score += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
            print(f'Correct predictions in batch: {test_score}\n')
            
            # calculate confusion matrix elements
            for i in range(args.batch_size):
              confusion_matrix[torch.argmax(label_batch[i, :])][torch.argmax(predicted_labels[i, :])] += 1

            n += len(label_batch)
        
        # print(f'\nn = {n}')

        print(f'Confusion Matrix:\n {confusion_matrix}')
        #Use %run not !python3 to get cm to display in collab
        plot_confusion_matrix(confusion_matrix)

        test_score = test_score / n
        print(f'Final test accuracy: {test_score}')

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()
import argparse
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from create_dataset import create_dataset
from Models import Models
import matplotlib.pyplot as plt


from Confusion_matrix_graphic import plot_confusion_matrix
from Confusion_matrix_graphic import num_to_instrument


def main():
    parser = argparse.ArgumentParser(description='Test the individual instrument identification model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--test_folder', type=str, default='audio/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', type=str,  default='billy_joel_cnn')
    parser.add_argument('--model_type', type=str, default='Basic_4_Layer_CNN')
    parser.add_argument('--num_mels', type=int, default=64)
    parser.add_argument('--num_fft', type=int, default=2048)
    parser.add_argument('--hop_len', type=int, default=1000)
    args = parser.parse_args()

    # Dataset and model paths
    drive_path = '/content/drive/MyDrive/DL_data/'
    # json_path_test = f'{drive_path}nsynth-test/examples.json'
    # audio_input_path_test = f'{drive_path}nsynth-test/{args.test_folder}'
    json_path_test = f'{drive_path}nsynth-valid/examples.json'
    audio_input_path_test = f'{drive_path}nsynth-valid/{args.test_folder}'
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
    test_dataset = create_dataset(audio_input_path_test, json_path_test, args.num_mels, args.num_fft, args.hop_len)

    # Create Data Loader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

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
            #print(f'img_batch:\n{img_batch}\nlabel_batch:\n{label_batch}')

            img_batch = img_batch.reshape(img_batch.shape[0], 1, img_batch.shape[1], img_batch.shape[2])
            #plt.imshow(img_batch)
            predicted_labels = model(img_batch)
            #print(f'predicted_labels: {predicted_labels}')

            test_score += (predicted_labels.argmax(axis=1) == label_batch.argmax(axis=1)).sum().item()
            #print(f'Correct predictions in batch: {test_score}\n')

            
            # calculate confusion matrix elements
            for i in range(len(label_batch)):
              confusion_matrix[torch.argmax(label_batch[i, :])][torch.argmax(predicted_labels[i, :])] += 1

            #print some examples from a random batch
            if n == int(torch.rand(1)*10):

                num_examples = 20
                #make random list of example index's
                examples_index = torch.Tensor(num_examples).random_(0,len(label_batch)) 
                fig = plt.figure(figsize=(22, 17))                               
                rows= 4
                columns = 5
                i = 1
                #for each index grab it's image and plot
                for index in examples_index:                 
                    one_img = img_batch[index.long(), 0, :, :]
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(one_img.cpu())
                    plt.axis('off')
                    #flip axis so freuency on bottom
                    ax = plt.gca()
                    ax.invert_yaxis()
                    # plot w labels and predicted labels as titles
                    plt.title(f'Predicted: {num_to_instrument(torch.argmax(predicted_labels[index.long(), :]))}   True: {num_to_instrument(torch.argmax(label_batch[index.long(), :]))}')
                    i += 1

                #add common labels
                fig.add_subplot(111, frame_on=False)
                plt.tick_params(labelcolor="none", bottom=False, left=False)
                plt.xlabel("Time")
                plt.ylabel("Frequency")                
                            
            n += len(label_batch)
        
        # print(f'\nn = {n}')
        label_counts = torch.sum(confusion_matrix, dim=1).reshape(len(confusion_matrix), 1)
        confusion_matrix /= label_counts
        # print(f'Confusion Matrix:\n {confusion_matrix}')
        #Use %run not !python3 to get cm to display in collab
        plot_confusion_matrix(confusion_matrix)

        test_score = test_score / n
        print(f'Final test accuracy: {test_score}')

        print(f'shape of image batch {img_batch.size()}')

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()

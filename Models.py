from torch import nn
from torch.nn.functional import softmax
import torchvision.models as models

# List of models to choose from. Currently in list:
#   * Basic 4 layer CNN
#   * AlexNet
#   * VGG13_BN
#   * ResNet18
#   * DenseNet121
class Models():
    def __init__(self, model_name: str):
        self.model_list = ['basic_4_layer_cnn', 'alex_net', 'res_net_18', 'dense_net_121', 'vgg13_bn']
        self.input_model = model_name
        self.num_output_classes = 11
        if self.input_model.lower() not in self.model_list:
            raise ValueError('Model list does not contain model "%s"' %(model_name))
    
    def choose_model(self):
        if self.input_model.lower() == 'basic_4_layer_cnn':
            model = Basic_4_Layer_CNN()
        elif self.input_model.lower() == 'alex_net':
            model = models.alexnet(False, False)
            model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_output_classes, bias=True)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        elif self.input_model.lower() == 'res_net_18':
            model = models.resnet18(False, False)
            model.fc = nn.Linear(in_features=512, out_features=self.num_output_classes, bias=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.input_model.lower() == 'dense_net_121':
            model = models.densenet121(False, False)
            model.classifier = nn.Linear(in_features=1024, out_features=11, bias=True)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.input_model.lower() == 'vgg13_bn':
            model = models.vgg13_bn(False, False)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_output_classes, bias=True)
        return model


class Basic_4_Layer_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(                                            # Dimension starts with 1 of 128 x 128
            #larger kernel CNN layers
            nn.Conv2d(1, 6, kernel_size = 5, padding = 2), nn.ReLU(),            #Dimension becomes 6 of 128 x 128
            nn.MaxPool2d(kernel_size = 2, stride =2),                            #Dimension now 6 of 64 x 64
            nn.Conv2d(6, 16, kernel_size=5, padding = 2), nn.ReLU(),             #Dimension now 16 of 64 x 64  
            nn.Conv2d(16, 16, kernel_size=5, padding = 2), nn.ReLU(),             #Dimension now 16 of 64 x 64
            nn.MaxPool2d(kernel_size=2, stride=2),                               #Dimension now 16 of 32 x 32
            #nn.Dropout(0.1),
            #smaller kernel CNN layers
            nn.Conv2d(16, 24, kernel_size=3, padding = 1), nn.ReLU(),            #Dimension now 24 of 32 x 32                                                                     #Dimension now 16 of
            nn.Conv2d(24, 24, kernel_size=3, padding = 1), nn.ReLU(),            #Dimension now 24 of 32 x 32
            #nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),                               #Dimension now 24 of 16 x 16
            nn.Conv2d(24, 48, kernel_size=3, padding = 1), nn.ReLU(),            #Dimension now 48 of 16 x 16                                                                     #Dimension now 16 of
            nn.Conv2d(48, 48, kernel_size=3, padding = 1), nn.ReLU(),            #Dimension now 48 of 16 x 16
            nn.Conv2d(48, 48, kernel_size=3, padding = 1), nn.ReLU(),            #Dimension now 48 of 16 x 16
            nn.MaxPool2d(kernel_size=2, stride=2),                               #Dimension now 48 of 8  x 8
            #nn.Dropout(0.2),
            #fully connected layers
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 2000), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2000, 300), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 11)
                                            # Because we have 11 output classes
        )

    def forward(self, x):
        return softmax(self.net(x), dim=-1)
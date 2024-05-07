import torch.nn as nn

from torchvision import models

def build_model(pretrained=True, fine_tune=True, num_classes=2):
# define a dictionary of available models
    available_models = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'vgg11': models.vgg11,
        'vgg16': models.vgg16,
        'densenet121': models.densenet121,
        'mobilenet_v2': models.mobilenet_v2
    }
    model_name= 'resnet50'
    model_ft = available_models[model_name]#(pretrained=True)

    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        #model = models.resnet34(pretrained=True)
        model = model_ft()

    else:
        print('[INFO]: Not loading pre-trained weights')
        model = model_ft()
        
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    num_ftrs = model.fc.in_features 
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

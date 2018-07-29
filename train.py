# Imports here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from PIL import Image

def DataSet_Values(args):
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder

    image_datasets = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    valid_datasets = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_datasets = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    return dataloaders,validloader,testloader,image_datasets,valid_datasets,test_datasets


def Train(model,dataloader,validloader,criterion,optimizer,epochs,print_every,device='cpu'):
    
    i = 0
    Training_Loss=0
    if device and torch.cuda.is_available():
        model.cuda()
    else:
        print('Go without cuda')
    for e in range(epochs):
        model.train()
        for Images,Labels in iter(dataloader):
            i += 1
            
            if device and torch.cuda.is_available():
                Images = Images.cuda()
                Labels = Labels.cuda()
            else:
                Images =Variable(Images)
                Labels =Variable(Labels)
            optimizer.zero_grad()
            Result = model.forward(Images)
            Loss_Value = criterion( Result, Labels)
            Loss_Value.backward()
            optimizer.step()
            Training_Loss += loss.data.item()
            if i % print_every == 0:
                model.eval()
                Accuracy=0
                Valid_Loss=0
                for Images, Labels in iter(validloader):
            
                    if device and torch.cuda.is_available():
                        Images =Images.cuda()
                        Labels = Labels.cuda()
                    else:
                        Images = Variable(Images)
                        Labels =  Variable(Labels)
                    with torch.no_grad():
                        Result = model.forward(Images)
                        Valid_Loss=criterion( Result, Labels).data.item()
                        Prob = torch.exp( Result).data
                        Equality = (Labels.data == Prob.max(1)[1])
                        Accuracy += Equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(Training_Loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(Valid_Loss/len(validloader)),
                      "Accuracy: {:.3f}".format(Accuracy/len(validloader)))
                running_loss = 0
                model.train()
    return model

def Model_Architecture(args):
    dataloaders=DataSet_Values(args)
    if args.arch=='vgg': 
        model=models.vgg16(pretrained=True)
    elif args.arch=='densenet':
        model=models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    Initial_Input = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(Initial_Input,512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512,args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units,102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.classifier = classifier

    if args.gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            print ("Inside GPU: "+ str(torch.cuda.is_available()))
        else:
            print("Inside CPU")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.lr)
    model = Train(model,dataloader,validloader,criterion,optimizer,epochs,print_every,device='cpu')

    model.class_to_idx=trainloader.dataset.class_to_idx
    model.epochs=args.epochs
    checkpoint={'input_size':[3,224,224],
                  'batch_size':trainloader.batch_size,
                  'output_size':102,
                  'arch':args.arch,
                  'state_dict':model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'epoch':model.epochs}
    torch.save(checkpoint,args.saved_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower Classifcation')
    parser.add_argument('--gpu', type=bool, default=False, help='CHECK GPU AVAILIBILTY')
    parser.add_argument('--arch', type=str, default='densenet', help='CHECK ARCHITECTURE OF PREDIFINED NETWORKS [available: densenet, vgg]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='LEARNING RATE')
    parser.add_argument('--hidden_units', type=int, default=100, help='HIDDEN UNITS FOR FC LAYERS')
    parser.add_argument('--epochs', type=int, default=15, help='EPOCHES')
    parser.add_argument('--data_dir', type=str, default='flowers', help='DATASET DIRECTORY')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='PATH OF SAVED MODEL')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    Model_Architecture(args)

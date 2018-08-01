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

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    Image_Size=256
    image.thumbnail(Image_Size, Image.ANTIALIAS)
    image=image.crop((128 - 112,128-112,128+112,128 + 112))
    My_Image = np.array(image)
    My_Image = My_Image/255
        
    X1=My_Image[:,:,0]
    X2=My_Image[:,:,1]
    X3=My_Image[:,:,2]
    
    X1=(X1-0.485)/(0.229) 
    X2=(X2-0.456)/(0.224)
    X3=(X3-0.406)/(0.225)
    
    My_Image[:,:,0]=X1
    My_Image[:,:,1]=X2
    My_Image[:,:,2]=X3
    
    My_Image = np.transpose(My_Image, (2,0,1))
    return My_Image


# TODO: load the checkpoint
def load_checkpoint(filepath):

    checkpoint_provided = torch.load(args.saved_model)
    if checkpoint_provided['arch'] == 'vgg':
        model = models.vgg16()  
        initial_input= model.classifier[0].in_features
    elif checkpoint_provided['arch'] == 'densenet':
        model = models.densenet121()
        initial_input= model.classifier.in_features
        
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(initial_input, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, 100)),                       
                              ('fc2', nn.Linear(100, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    if args.gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            print ("Using GPU")
        else:
            print("Using CPU")
    loaded_model, class_to_idx = load_checkpoint('my_checkpoint.pth')
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class

def predict(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    My_Image = torch.FloatTensor([process_image(Image.open(image_path))])
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    if args.gpu and torch.cuda.is_available():
        Result = model.forward(Variable(My_Image.cuda()))
    else:
        Result = model.forward(Variable(My_Image))
    PS = torch.exp(Result).data.numpy()[0]
    Top_Index = np.argsort(ps)[-topk:][::-1] 
    Top_Class = [idx_to_class[x] for x in Top_Index]
    Probability = PS[Top_Index]

    print("Probability is : {} \n Top class is : {}".format(Probability, Top_Class))
    return Probability, Top_Class

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Flower Classifcation')
    parser.add_argument('--gpu', type=bool, default=False, help='CHECK GPU AVAILIBILTY')
    parser.add_argument('--image_path', type=str, help='IMAGE PATH')
    parser.add_argument('--hidden_units', type=int, default=100, help='HIDDEN UNITS FOR FC LAYERS')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='PATH OF SAVED MODEL')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='PATH OF MAPPING ARCHITECTURE FROM COTEGORY TO NAME')
    parser.add_argument('--topk', type=int, default=5, help='TOP K PROBABILITIES')
    args = parser.parse_args()

    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    Top_Probability, Top_class = predict(args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes: ', Top_Class)
    print ('Class Names: ')
    [print(cat_to_name[x]) for x in Top_Class]
    print('Predicted Probability: ', Top_Probability)

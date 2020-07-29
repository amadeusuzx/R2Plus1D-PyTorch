import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset, VideoDataset1M
from network import R2Plus1DClassifier

import sys

# Use GPU if available else revert to CPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")



def train_model(num_classes, directory, layer_sizes=[2, 2, 2, 2], num_epochs=45, save=True, path="model_data.pth.tar"):
    from torch_videovision.torchvideotransforms import video_transforms, volume_transforms
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """


    # initalize the ResNet 18 version of this model
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)

    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(),lr=0.01)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # prepare the dataloaders into a dict
    video_transform_list = [video_transforms.RandomCrop((25,50)),
            video_transforms.RandomRotation(30),
            video_transforms.RandomHorizontalFlip(0.3),
            video_transforms.ColorJitter(0.5,0.5,0.5,0.5)]
    transforms = video_transforms.Compose(video_transform_list)

    train_set = VideoDataset(directory,transforms=transforms)
    val_set = VideoDataset(directory,mode = 'val')

    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
    # IF training on Kinetics-600 and require exactly a million samples each epoch, 
    # import VideoDataset1M and uncomment the following
    # train_dataloader = DataLoader(VideoDataset1M(directory), batch_size=32, num_workers=4)
    val_dataloader = DataLoader(val_set, batch_size=16, num_workers=8)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()


            for inputs, labels in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")
        if save:
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': epoch_acc,
            'opt_dict': optimizer.state_dict(),
            }, path)
    # save the model if save=True
    

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")

def test_model(num_classes, directory, layer_sizes=[2, 2, 2, 2], path="model_data.pth.tar"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """


    # initalize the ResNet 18 version of this model 
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification

    test_dataloader = DataLoader(VideoDataset(directory, mode='test'), batch_size=1, num_workers=0)

    dataset_size =len(test_dataloader.dataset)

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    # check if there was a previously saved checkpoint
    checkpoint = torch.load(path)
    print("Reloading from checkpoint")

    # restores the model and optimizer state_dicts
    model.load_state_dict(checkpoint['state_dict'])

    running_loss = 0.0
    running_corrects = 0
    model.eval()


    for inputs, labels in test_dataloader:
        # move inputs and labels to the device the training is taking place on
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  
        loss = criterion(outputs, labels)


        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f"test Loss: {epoch_loss} Acc: {epoch_acc}")
    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Test complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
def test_file(num_classes, fname, layer_sizes=[2, 2, 2, 2], path="model_data.pth.tar"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """


    # initalize the ResNet 18 version of this model 
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)
    test = VideoDataset('./lips/', mode='val')
    with open("test.txt","w") as t:
        for te in test.fnames:
            t.write(te+"\n")
    
    checkpoint = torch.load(path,map_location='cpu')
    print("Reloading from checkpoint")
    # restores the model and optimizer state_dicts
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_file = test.loadvideo(fname)
    input_file = test.normalize(input_file)
    input_file = torch.tensor(input_file)
    input_file = input_file.reshape([1]+[s for s in input_file.shape])
    outputs = model(input_file)
    _, preds = torch.max(outputs, 1)  
    #commands = sorted(["black","cancel","centeralign","copy","large","medium","newslide","paste","red","textbox"])
    commands = sorted(["open",
            "close",
            "press",
            "release",
            "scroll_up",
            "scroll_down",
            "task_switch",
    ])
    for p in preds:
        print(commands[p])
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_model(int(sys.argv[2]),sys.argv[3],num_epochs=int(sys.argv[4]),path=sys.argv[5])
    elif sys.argv[1] == "test":
        test_model(int(sys.argv[2]),sys.argv[3])
    elif sys.argv[1] == "test_file":
        test_file(int(sys.argv[2]),sys.argv[3],path = sys.argv[4])
import torch
import os
from datasets import dataset
import pandas


def read_dataset(input_size, batch_size, root, set):
    if set == 'CUB':
        print('Loading CUB trainset')
        trainset = dataset.CUB(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading CUB testset')
        testset = dataset.CUB(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'CAR':
        print('Loading car trainset')
        trainset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading car testset')
        testset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Aircraft':
        print('Loading Aircraft trainset')
        trainset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading Aircraft testset')
        testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'mango':
        print('Loading Mango trainset')
        traindf = pandas.read_csv(os.path.join(root, 'filteredTrain.csv'))
        trainfolder = '/content/drive/Shared drives/Mango_Robot_vision/yun-shuo/MangoSegDetectron2/transformedTrain'
        traindataset = dataset.Dataframedataset(trainfolder, traindf, input_size)
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                                  drop_last=False)
        print('Loading Mango testset')
        testdf = pandas.read_csv(os.path.join(root, 'dev.csv'))
        testfolder = '/content/drive/Shared drives/Mango_Robot_vision/yun-shuo/MangoSegDetectron2/transformedDev'
        testset = dataset.Dataframedataset(testfolder, testdf, input_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8,
                                                 drop_last=False)
    else:
        print('Please choose supported dataset')

    return trainloader, testloader

import creating_triples as ct
import Loss_Function as lf
from Model import Model
import pickle
from liveplot import create_plot

from torch import Tensor, stack
import torch
from torch.optim.adam import Adam
from torch import margin_ranking_loss
from collections import Counter
import numpy as np

import pickle

def embedding_caption():
        return np.load("embeddings.npy").item()

def getFeatures():
    with open("resnet18_features_train.pkl", mode="rb") as opened_file:
        rf =  pickle.load(opened_file)
    return rf

def preTraining():
    '''
    Creates the model, data and optimizor for training

    -----
    Returns
    -----
    data - tuple with caption_ids, good_image and bad_image
    optimizer - Adam optimizer from  mynn
    model - traiing model
    '''
    model = Model()
    data = ct.create_triples(100).reshape(-1,3)
    optim = Adam(model.parameters())
    return data, optim, model

def training():

    plotter, fig, ax = create_plot(["loss", "accuracy"], refresh=5)
    arr_of_data = np.array(data)
    batch_size = 100

    for epoch_cnt in range(10):
        idxs = np.arange(len(data))  # -> array([0, 1, ..., 9999])
        np.random.shuffle(idxs)

        for batch_cnt in range(0, len(data)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            batch = data[batch_indices]  # random batch of our training data
            text = []
            good_image = []
            bad_image = []
            for tuple_of_data in batch:
                try:
                    embedded_captions[tuple_of_data[0]]
                    rf[tuple_of_data[1]]
                    rf[tuple_of_data[2]]
                except KeyError:
                    continue
                text.append(Tensor(embedded_captions[tuple_of_data[0]]).view(1,-1))
                good_image.append(model(rf[tuple_of_data[1]]).view(-1,1))
                bad_image.append(model(rf[tuple_of_data[2]]).view(-1,1))
            text = normalize(stack(text))
            good_image = normalize(stack(good_image))
            bad_image = normalize(stack(bad_image))
            loss_batch = lf.loss((text, good_image, bad_image), 0.2)#pick a loss later
            print(loss_batch)
            loss = torch.sum(loss_batch)/len(loss_batch)
            loss.backward()
            acc = lf.accuracy(loss_batch)
            optim.step()
            optim.zero_grad()
            plotter.set_train_batch({"loss" : loss.item(),
                                     "accuracy" : acc},
                                     batch_size=batch_size)
        # this tells liveplot to plot the epoch-level train/test statistics :)
        plotter.plot_train_epoch()
        plotter.plot_test_epoch()
def normalize(data):
    mag = torch.sqrt(torch.sum(data**2, dim = 1, keepdim = True))
    return data/mag
def query():
    return top_images
def find_top_k():
    pass
def display():
    pass

embedded_captions = embedding_caption()
rf = getFeatures()
data, optim, model = preTraining()

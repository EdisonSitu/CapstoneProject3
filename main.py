import creating_triples as ct
import Loss_Function as lf
from Model import Model
import pickle
from liveplot import create_plot

from torch.optim.adam import Adam
from mynn.initializers.glorot_normal import glorot_normal
from torch import margin_ranking_loss, sum, sqrt
from collections import Counter
import numpy as np
import mygrad as mg

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
    data = ct.create_triples(10).reshape(-1,3)
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
                text.append(nn.Tensor(embedded_captions[tuple_of_data[0]]).view(1,-1))
                good_image.append(model(rf[tuple_of_data[1]]).data.view(-1,1))
                bad_image.append(model(rf[tuple_of_data[2]].data.view(-1,1)))
            text = normalize(Tensor(text))
            good_image = normalize(Tensor(good_image))
            bad_image = normalize(Tensor(bad_image))
            loss = lf.loss((text, good_image, bad_image), 0)#pick a loss later
            loss.backward()
            acc = lf.accuracy(loss)
            optim.step()
            loss.zero_grad()
            print(loss)
            print(acc)
            plotter.set_train_batch({"loss" : loss.item(),
                                     "accuracy" : acc},
                                     batch_size=batch_size)

        # this tells liveplot to plot the epoch-level train/test statistics :)
        plotter.plot_train_epoch()
        plotter.plot_test_epoch()
def normalize(data):
    mag = sqrt(sum(data**2))
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

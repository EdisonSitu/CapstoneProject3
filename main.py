import creating_triples as ct
import Loss_Function as lf
import word_embedding as we
from Model import Model
from database import Database
from display import display

import pickle
from liveplot import create_plot

from torch import Tensor, stack, matmul
import torch
from torch.optim.sgd import SGD
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
    data = ct.create_triples(300000, 25).reshape(-1,3)
    optim = SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    return data, optim, model

def training():
    '''
    Initiates training
    '''
    #creates data/parameters
    embedded_captions = embedding_caption()
    rf = getFeatures()
    data, optim, model = preTraining()
    plotter, fig, ax = create_plot(["loss", "accuracy"], refresh=5)
    arr_of_data = np.array(data)
    batch_size = 100
    idxs = np.arange(len(arr_of_data))  # -> array([0, 1, ..., 9999])
    np.random.shuffle(idxs)
    xtrain = arr_of_data[idxs[:len(idxs) * 9 //10]]
    xtest = arr_of_data[idxs[len(idxs) * 9 //10:]]
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
            loss_batch = lf.loss((text, good_image, bad_image), 0.1)#pick a loss later
            loss = torch.sum(loss_batch)/len(loss_batch)
            loss.backward()
            acc = lf.accuracy(loss_batch)
            optim.step()
            model.zero_grad()
            plotter.set_train_batch({"loss" : loss.item(),
                                     "accuracy" : acc},
                                     batch_size=batch_size)
        # this tells liveplot to plot the epoch-level train/test statistics :)
        plotter.plot_train_epoch()
        plotter.plot_test_epoch()

    #adds to database
    databasing(model, rf)
    #saves model
    torch.save(model.state_dict(), "model_parameters.txt")
def normalize(data):
    '''
    Normalizes the data to zero - center and remove std
    '''
    mag = torch.sqrt(torch.sum(data**2, dim = 1, keepdim = True))
    return data/mag

def databasing(model, rf):
    '''
    Adding image (embedded) into the database given the model
    '''
    image_ids = ct.getImage_ids()
    for image_id in image_ids:
        try:
            database.put(image_id,model(rf[image_id]))
        except KeyError:
            continue
    database.save()
#given an image, returns similar ones
def query(word, k):
    '''
    Get an image data and returns top k images that are similar
    '''
    print("start query")
    e_word = Tensor(we.embed_caption(word)[np.newaxis,:])
    print("normalizing")
    e_word = normalize(e_word)
    top_ids = find_top_k(e_word, k)
    display(top_ids)
def find_top_k(e_word, k):
    '''
    Finds the top k images given embedded caption
    '''
    list_of_comparisons = []
    #get image_ids
    image_ids = np.array(ct.getImage_ids(database.dictionary))
    list_of_embeddings = stack([database.get(image_id) for image_id in image_ids])
    list_of_comparisons = matmul(list_of_embeddings, e_word.view(-1,1)).data.numpy()
    print(list_of_comparisons.shape)
    sorted_indexs = np.argsort(list_of_comparisons.flatten())[::-1]
    print(image_ids[sorted_indexs[:k]])
    return image_ids[sorted_indexs[:k]]
database = Database('database.txt')
database.load()
model = Model()
model.load_state_dict(torch.load("model_parameters.txt"))

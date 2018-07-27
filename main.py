import creating_triples
import Loss_Function
import Model

from mynn.optimizers.adam import Adam
from mynn.initializers.glorot_normal import glorot_normal
from torch import margin_ranking_loss
from collections import Counter

model = Model()
optim = Adam(model.parameters)

def embedding_caption():
    return e_cap #dictionary
def embedding_images():
    return e_img, model

def training(list_of_data, model):

    arr_of_data = np.array(list_of_data)
    batch_size = 100

    for epoch_cnt in range(10):
        idxs = np.arange(len(list_of_data))  # -> array([0, 1, ..., 9999])
        np.random.shuffle(idxs)

        for batch_cnt in range(0, len(list_of_data)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            batch = arr_of_data[batch_indices]  # random batch of our training data
            text = []
            for tuple_of_data in arr_of_data:
                text.append(dictionary[tuple_of_data[0]]) #whatever dictionary is named
            text = normalize(mg.Tensor(text))
            good_image = model(batch[1])
            good_image = normalize(good_image)
            bad_image = model(batch[2])
            bad_image = normalize(image)
            loss = loss(text, good_image, bad_image, 0)#pick a loss later
            loss.backwards()
            acc = accuracy(loss)
            optim.step()
            loss.null_gradient()
            plotter.set_train_batch({"loss" : loss.item(),
                                     "accuracy" : acc},
                                     batch_size=batch_size)

        # this tells liveplot to plot the epoch-level train/test statistics :)
        plotter.plot_train_epoch()
        plotter.plot_test_epoch()

def normalize(data):
    mag = mg.sqrt(mg.sum(data**2))
    return data/mag
def query():
    return top_images
def find_top_k():
    pass
def display():
    pass

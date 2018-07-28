from torch import margin_ranking_loss, Tensor
import mygrad as mg
import numpy as np

def loss(tuple_data, margin):
    '''
    Determines the loss of the data (caption compared to good vs caption compared to bad). Returns max(0, margin - good + bad)

    ------
    Parameters
    ------
    tuple_data (np.array(int), np.array(int), np.array(int))
        Data containing embedded text, embedded image data (classified as "good"), and another embedded image data (classified as "bad")
    margin (int)
        Margin that (good-bad) must exceed

    -----
    Returns
    -----
    loss (int)
        An integer that represents the loss
    '''
    text, good_image, bad_image = tuple_data
    print(text.shape)
    print(good_image.shape)
    print(bad_image.shape)
    x1 = torch.mm(text, good_image).reshape(-1)
    x2 = torch.mm(text, bad_image).reshape(-1)
    print(type(x1))
    print(type(x2))
    return margin_ranking_loss(x1, x2, Tensor(np.ones(len(x1))), margin = margin, reduce= False)


def accuracy(loss_batch):
    '''
    Determines the accuracy of a batch of losses

    -----
    Parameters
    -----
    loss_batch tuple(ints)
        a list of all the losses

    -----
    Returns
    -----
    The average success of the batch
    '''
    return np.sum(np.sign(loss_batch))/len(loss_batch)

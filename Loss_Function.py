from torch import margin_ranking_loss

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
    x1 = mg.dot(text, good_image)/(mg.abs(text)*mg.abs(good_image))
    x2 = mg.dot(text, bad_image)/(mg.abs(text)*mg.abs(bad_image))
    loss_obj =  margin_ranking_loss(margin = margin, reduce= False)
    return loss_obj(x1, x2, y)
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

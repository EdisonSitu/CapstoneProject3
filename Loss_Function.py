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
    x1 = np.dot(text, good_image)
    x2 = np.dot(text, bad_image)
    return margin_ranking_loss(x1, x2, 1, margin)
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

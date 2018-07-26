def loss(text, good, bad, margin):
    x1 = np.dot(text, good_image)
    x2 = np.dot(text, bad_image)
    return margin_ranking_loss(x1, x2, 1, margin)

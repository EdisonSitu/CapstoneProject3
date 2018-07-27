""" STUFF TO DO:
    - create function to query database and return top k images
    - create function that finds top k similar images to a query image
"""
import re, string
import numpy as np
import pickle
from gensim.models.keyedvectors import KeyedVectors

vectors = # open database as a numpy.ndarray
mappings = # open database as a dictionary
idfs = np.load("idfs.npy").item()
path = "../week3_student/word_embeddings/glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)
with open("resnet18_features_train.pkl", mode="rb") as opened_file:
    res_feat = pickle.load(opened_file)

img_ids = list(res_feat.keys())
original = np.array([res_feat[i].numpy() for i in img_ids]).reshape(82612, 512)

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(corpus):
    return punc_regex.sub('', corpus)

def text_query (text, k=1):
    """ Gets k images based on a text query.

        Parameters
        ----------
        k : int
            The number of images to return.

        text : str
            The caption to query.

        Returns
        -------
        list, len(k)
            Returns a list of length k of image URLs to display.
        """
    x = np.zeros(50)
    count = 0
    tokens = strip_punc(text).lower().split()
    for word in tokens:
        try:
            x += glove.get_vector(word) * idfs[word]
            count += 1
        except Exception as e:
            continue
    embed = x / count

    diffs = np.dot(vectors, embed)
    unsorted_diff_indices = np.argsort(diffs[::-1])
    diffs[::-1].sort()
    urls = list()
    for i in unsorted_diff_indices[:k]:
        urls.append(mappings[i])
    return urls

def img_query (img, semantic=True, k=1):
    """ Gets k images based on a image query.

        Parameters
        ----------
        k : int
            The number of images to return.

        img : np.ndarray, shape (50,)
            The image vector of an image. Shape is (512,) if the image vector is
            the

        semantic : boolean
            Determines whether the image should be mapped based off of the
            original image space or the learned semantic space

        Returns
        -------
        list, len(k)
            Returns a list of length k of image URLs to display.
        """
    if semantic:
        diffs = np.dot(vectors, img)
    else:
        diffs = np.dot(original, img)
    unsorted_diff_indices = np.argsort(diffs[::-1])
    diffs[::-1].sort()
    urls = list()
    if semantic:
        for i in unsorted_diff_indices[:k]:
            urls.append(mappings[i])
    else:
        for i in unsorted_diff_indices[:k]:
            urls.append(img_ids[i])
    return urls

import json
json_data = open("captions_train2014.json").read()
data = json.loads(json_data)
annotations = data["annotations"]
import re, string
import numpy as np
from collections import Counter
import pickle
from gensim.models.keyedvectors import KeyedVectors

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))


def strip_punc(corpus):
    '''
    Removes Punctuations
    '''
    return punc_regex.sub('', corpus).lower()
def get_stop_words():
    '''
    Returns stopwords
    '''
    with open("stopwords.txt", 'r') as r:
        stops = []
        for line in r:
            stops += [i.strip() for i in line.split('\t')]
    return stops
#get idf and its setup
def to_counter(doc):
    """
    Produce word-count of document, removing all punctuation
    and removing all punctuation.

    Parameters
    ----------
    doc : str

    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    return Counter(strip_punc(doc).lower().split())
def to_bag(counters, k=None, stop_words=None):
    """
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`

    Parameters
    ----------
    counters : Iterable[Iterable[str]]

    k : Optional[int]
        If specified, only the top-k words are returned

    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the bag
    """
    bag = Counter()
    for counter in counters:
        bag.update(counter)

    if stop_words is not None:
        for word in set(stop_words):
            bag.pop(word, None)  # if word not in bag, return None
    return sorted(i for i,j in bag.most_common(k))

def to_idf(bag, counters):
    """
    Given the bag-of-words, and the word-counts for each document, computes
    the inverse document-frequency (IDF) for each term in the bag.

    Parameters
    ----------
    bag : Sequence[str]
        Ordered list of words that we care about

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.

    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `bag`, storing
        the IDF for each term `t`:
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of
        documents in which the term `t` occurs.
    """
    N = len(counters)
    all_occurence_of_words = np.array([word for counter in counters for word in counter])
    all_words = {word for counter in counters for word in counter.keys()}
    all_words = sorted(list(all_words & set(bag)))
    nt = [len(np.where(all_occurence_of_words == word)[0]) for word in all_words]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)


def getBigBag():
    '''
    Gets the big bag of words
    '''
    count_list = []
    for dictionary in annotations:
        count_list.append(to_counter(dictionary["caption"]))
    bag = to_bag(count_list, stop_words=get_stop_words())
    return bag
#get big bag
def idfs_in_bulk():
    '''
    Calculates IDF in bulk
    '''
    count_list = []
    for dictionary in annotations:
        count_list.append(to_counter(dictionary["caption"]))
    bag = to_bag(count_list, stop_words=stops)
    idf = to_idf(bag, count_list)
    idfs = dict(zip(bag, idf.tolist()))
    with open("idfs.pkl", mode="wb") as opened_file:
        pickle.dump(idfs, opened_file)
#Get glove
def getGlove():
    '''
    Obtains glove
    '''
    path = "./glove.6B.50d.txt.w2v"
    return KeyedVectors.load_word2vec_format(path, binary=False)

def embed_caption(caption):
    idfs = []
    words = strip_punc(caption).split()
    idfs_all = np.load("idfs.npy").item()
    glove = getGlove()
    for word in words:
        print(word)
        try:
            idfs.append(idfs_all[word])
        except KeyError:
            continue
    idfs = np.array(idfs)
    count = 0
    x = np.zeros(50)
    for i, word in enumerate(words):
        try:
            x += glove.get_vector(word) * idfs[i]
            count += 1
            print(x)
        except Exception as e:
            continue
    if count != 0:
        embed = x / count
    return x

def embedding_in_bulk():
    idfs = idfs_in_bulk()
    caption_embeddings = {}
    for cap_dict in annotations:
        x = np.zeros(50)
        count = 0
        for word in cap_dict["caption"].split():
            try:
                x += glove.get_vector(word.lower()) * idfs[word.lower()]
                count += 1
            except Exception as e:
                continue
        if count != 0:
            embed = x / count
            caption_embeddings[cap_dict["id"]] = embed
    np.save("embeddings.npy", caption_embeddings)

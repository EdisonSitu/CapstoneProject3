""" STUFF TO DO:
    - method to create k triples for a caption
"""
import pickle
import json
import numpy as np

with open("resnet18_features_train.pkl", mode="rb") as opened_file:
    res_feat = pickle.load(opened_file)

json_data = open("captions_train2014.json").read()
data = json.loads(json_data)

def create_triples (length, k=1):
    """ Creates an array of triples for each caption ID.

        Parameters
        ----------
        k : int
            The number of triples to create per caption.

        Returns
        -------
        np.ndarray, shape=(414113, k, 3)
            Creates an array of k triples for each caption.
        """
    idxs = np.arange(len(data["annotations"]))
    np.random.shuffle(idxs)
    caption_ids = np.tile([data["annotations"][i]["id"] for i in idxs[:length]], (k, 1)).T
    image_ids = np.tile([data["annotations"][i]["image_id"] for i in idxs[:length]], (k, 1))
    all_ids = np.array([data["annotations"][i]["image_id"] for i in range(len(data["annotations"]))])
    triples_array = np.stack((caption_ids, image_ids.T, np.random.choice(all_ids, size=(length, k))), axis=-1)
    return triples_array
def unzip(pairs):
    """Splits list of pairs (tuples) into separate lists.

    Example: pairs = [("a", 1), ("b", 2)] --> ["a", "b"] and [1, 2]

    This should look familiar from our review back at the beginning of week 1
    :)
    """
    return tuple(zip(*pairs))
def getImage_ids(dictionary):
    return list(dictionary.keys())

def getImage_ids_no_dictionary():
    return unzip(resfeat.items())[0]

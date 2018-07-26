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

def create_triples (k=1):
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
    caption_ids = np.tile([data["annotations"][i]["id"] for i in range(len(data["annotations"]))], (k, 1)).T
    image_ids = np.tile([data["annotations"][i]["image_id"] for i in range(len(data["annotations"]))], (k, 1)).T
    triples_array = np.stack((caption_ids, image_ids, np.random.choice(image_ids[0], size=(414113, k))), axis=-1)
    return triples_array

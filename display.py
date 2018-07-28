import json
json_data = open("captions_train2014.json").read()
data = json.loads(json_data)
images = data["images"]
image_ids = [images[i]["id"] for i in range(len(images))]
coco_urls = [images[i]["coco_url"] for i in range(len(images))]

import matplotlib.pyplot as plt
from skimage import io

def display(ids):
    '''
    Parameters:
        ids - list
    '''
    for id in ids:
        link = coco_urls[image_ids.index(id)]
        io.imshow(io.imread(link))
        io.show()
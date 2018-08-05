import json
json_data = open("captions_train2014.json").read()
data = json.loads(json_data)
images = data["images"]
image_ids = [images[i]["id"] for i in range(len(images))]
coco_urls = [images[i]["coco_url"] for i in range(len(images))]
data = dict(zip(image_ids, coco_urls))
import matplotlib.pyplot as plt
from skimage import io
def display(ids):
    '''
    Parameters:
        ids - list
    '''
    fig, axs = plt.subplots(nrows = len(ids))
    for i, id in enumerate(ids):
        link = data[id]
        axs[i].imshow(io.imread(link))
        plt.axis('off')

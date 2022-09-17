import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def process_label(labels):
    """
    Blackboard/Whiteboard class in Object365 is class 101. We have three classes in own custom dataset.
    Rearrange label into similar format as Object365
    """
    labels[np.where(labels == 0)] = 101
    labels[np.where(labels == 2)] = 101

    return labels


def process_textfile(line):
    """
    Clean the txt annotations and store as numpy array
    """
    object_annot = line.strip().split(' ')
    return np.array(object_annot).astype('float64')


class WhiteboardTestDataloader(Dataset):

    def __init__(self):
        self.img_files = glob.glob('Whiteboards-11/test/Images/*.jpg')
        self.annot_files = glob.glob('Whiteboards-11/test/Labels/*.txt')
        self.image_width = 640
        self.image_height = 640

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img_path = self.img_files[item]
        annot_path = self.annot_files[item]
        img = Image.open(img_path)

        # Read annotation data
        annotations = [process_textfile(x) for x in open(annot_path, "r").readlines()]
        annotations = np.stack(annotations, axis=0)
        annotations[:, 1:] = self.process_annotation(annotations[:, 1:])
        annotations[:, 0] = process_label(annotations[:, 0])
        annotations = annotations[np.where(annotations[:,0] == 101)]
        # annotations = torch.from_numpy(annotations).type(torch.FloatTensor)

        return np.array(img), annotations

    def process_annotation(self, annotation):
        """
        converting annotations in the form of center point and width height into top-left and bottom-right
        coordinates of Bounding box
        """
        x = annotation[:, 0]
        y = annotation[:, 1]
        w = annotation[:, 2]
        h = annotation[:, 3]

        w_half = np.ceil(self.image_width / 2 * w)
        h_half = np.ceil(self.image_height / 2 * h)

        # x = int(x * self.image_width)
        x = np.ceil(x * self.image_width)
        y = np.ceil(y * self.image_height)
        x1 = x - w_half
        y1 = y - h_half
        x2 = x + w_half
        y2 = y + h_half
        x1[np.where(x1 < 0)] = 0
        y1[np.where(y1 < 0)] = 0
        x2[np.where(x2 > self.image_width)] = self.image_width
        y2[np.where(y2 > self.image_height)] = self.image_height

        # Rearrange annotations in form of the result returned from the model
        annotation_processed = np.stack([x1, y1, x2, y2], axis=1)

        return annotation_processed


if __name__ == "__main__":
    test_dl = WhiteboardTestDataloader()

    test_dl.__getitem__(0)

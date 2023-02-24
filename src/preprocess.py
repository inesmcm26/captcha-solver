import numpy as np
import os
import cv2
from imutils import paths, resize
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pickle
from char_detection import simple_preprocess, img2boxes

def load_chars(img_folder):
    """
    Get the images and labels from the folder
    """
    data = []
    labels = []
    # loop over the input images
    for i, image_file in enumerate(paths.list_images(img_folder)):
        
        # Load the image and convert it to grayscale
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 20x20 pixel box
        img = resize_to_fit(img, 20, 20)

        # Add a third channel dimension to the img to make Keras work
        img = np.expand_dims(img, axis=-1)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels)

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    """
    (h, w) = image.shape[:2]

    # resize along the largest axis
    if w > h:
        image = resize(image, width=width)
    else:
        image = resize(image, height=height)

    # padding values
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height)) # only to avoid rounding issues

    return image

def one_hot_labels(y_train, y_val, y_test, save_path):
    """
    One hot encode the labels
    Saves the label binarizer for later decode the predictions
    """
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_val = lb.transform(y_val)
    y_test = lb.transform(y_test)

    with open(save_path, 'wb') as f:
        pickle.dump(lb, f)

    return y_train, y_val, y_test





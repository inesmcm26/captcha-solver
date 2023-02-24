import numpy as np
import matplotlib.pyplot as plt

def img_predict(model, lb, chars):
    """
    For a list of character images, predicts the characters
    """
    predictions = []


    for char_img in chars:

        char_img = np.expand_dims(char_img, axis=0)

        pred = model.predict(char_img)

        char = lb.inverse_transform(pred)[0]

        predictions.append(char)

    return "".join(predictions)
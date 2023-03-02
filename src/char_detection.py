import os
import cv2
import imutils
import glob
import numpy as np
from preprocess import resize_to_fit

def simple_imgs2chars(input_folder, output_folder):
    """
    Detects the characters in the simple captcha images
    Each character is saved in a separate file in gray scale
    """
    # Get a list of all the captcha images we need to process
    images = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    # loop over the image paths
    for (i, img_path) in enumerate(images):
        print("[INFO] processing image {}/{}".format(i + 1, len(images)))

        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(img_path)
        img_text = os.path.splitext(filename)[0]

        img, gray = simple_thresh(img_path)

        char_bound_boxes = img2boxes(img, lambda w, h: w / h > 1.25, num_chars = 4, cv2_chain = cv2.CHAIN_APPROX_SIMPLE)
        
        if char_bound_boxes is None:
            continue

        for char_bound_box, char in zip(char_bound_boxes, img_text):
            counts = crop_and_save(char_bound_box, char, gray, output_folder, counts)

def hard_imgs2char(input_folder, output_folder):
    """
    Detects the characters in the hard captcha images
    Each character is saved in a separate file in gray scale
    """
    # Get a list of all the captcha images we need to process
    images = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    for (i, img_path) in enumerate(images):
        print(f"INFO: processing image {i+1}/{len(images)}")

        # Extract name of the file since it contains the captcha characters
        filename = os.path.basename(img_path)
        img_text = os.path.splitext(filename)[0]

        img, gray = complex_thresh(img_path)

        char_bound_boxes = img2boxes(img, lambda w, h: (((w / h) > 1.35) and (w > 22)) or (h > 28), num_chars = 10, cv2_chain = cv2.CHAIN_APPROX_NONE)

        if char_bound_boxes is None:
            continue

        for char_bound_box, char in zip(char_bound_boxes, img_text):
            counts = crop_and_save(char_bound_box, char, gray, output_folder, counts)

def simple_thresh(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray

def complex_thresh(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # threshold the image with a costum mask because it is the best way to process this specific images
    lower = np.array([220,220,220])
    upper = np.array([255,255,255])
    my_mask = cv2.inRange(img, lower, upper)
    img = cv2.threshold(my_mask, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray


def img2boxes(img, conjoined_condition, num_chars, cv2_chain):
    """
    Find the bounding boxes of the characters in the image
    """
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2_chain)
    contours = contours[1] if imutils.is_cv3() else contours[0]

    char_bound_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if (num_chars == 10) and  (area < 90):
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        if conjoined_condition(w, h):
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            char_bound_boxes.append((x, y, half_width, h))
            char_bound_boxes.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            char_bound_boxes.append((x, y, w, h))

    if len(char_bound_boxes) != num_chars:
        return None

    char_bound_boxes = sorted(char_bound_boxes, key=lambda x: x[0])

    return char_bound_boxes

def find_chars(img_path):
    """
    Detects the characters in the captcha image
    """

    img, gray = simple_thresh(img_path)

    char_regions = img2boxes(img, lambda w, h: w / h > 1.25, num_chars = 4, cv2_chain = cv2.CHAIN_APPROX_SIMPLE)

    if char_regions is None:
        return None
    
    chars = []

    for char_region in char_regions:
        (x, y, w, h) = char_region
        
        char_img = gray[y-2:y+h+2, x-2:x+w+2]

        char_img = resize_to_fit(char_img, 20, 20)

        char_img = np.expand_dims(char_img, axis=-1)

        char_img = char_img / 255

        chars.append(char_img)

        gray = cv2.rectangle(gray, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)

    return gray, chars


def crop_and_save(char_bound_box, char, img, out_folder, counts):
    """
    Crop the character from the image and saves it as a new image
    """
    x, y, w, h = char_bound_box
    
    # get the letter from the original image with a 2 pixels margin around the edges
    char_img = img[y-2:y+h+2, x-2:x+w+2]

    save_path = os.path.join(out_folder, char)

    # if the output directory does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # write the letter image to a file
    count = counts.get(char, 1)
    p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    cv2.imwrite(p, char_img)

    counts[char] = count + 1

    return counts
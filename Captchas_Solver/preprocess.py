import os
import cv2
import imutils
import glob
import numpy as np

def simple_captcha_preprocess(input_folder, output_folder):
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

        img, gray = simple_preprocess(img_path)

        char_regions = get_countours(img, lambda w, h: w / h > 1.25, num_chars = 4, cv2_chain = cv2.CHAIN_APPROX_SIMPLE)
        
        if char_regions is None:
            continue

        # # find the contours (continuous blobs of pixels) the image
        # contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # Hack for compatibility with different OpenCV versions
        # contours = contours[1] if imutils.is_cv3() else contours[0]

        # letter_image_regions = []

        # # Now we can loop through each of the four contours and extract the letter
        # # inside of each one
        # for contour in contours:
        #     # Get the rectangle that contains the contour
        #     (x, y, w, h) = cv2.boundingRect(contour)

        #     # Compare the width and height of the contour to detect letters that
        #     # are conjoined into one chunk
        #     if w / h > 1.25:
        #         # This contour is too wide to be a single letter!
        #         # Split it in half into two letter regions!
        #         half_width = int(w / 2)
        #         letter_image_regions.append((x, y, half_width, h))
        #         letter_image_regions.append((x + half_width, y, half_width, h))
        #     else:
        #         # This is a normal letter by itself
        #         letter_image_regions.append((x, y, w, h))

        # # If we found more or less than 4 letters in the captcha, our letter extraction
        # # didn't work correcly. Skip the image instead of saving bad training data!
        # if len(letter_image_regions) != 4:
        #     continue

        # # Sort the detected letter images based on the x coordinate to make sure
        # # we are processing them from left-to-right so we match the right image
        # # with the right letter
        # letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # # Save out each letter as a single image
        # for letter_bounding_box, letter_text in zip(letter_image_regions, img_text):
        #     # Grab the coordinates of the letter in the image
        #     x, y, w, h = letter_bounding_box

        #     # Extract the letter from the original image with a 2-pixel margin around the edge
        #     letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        #     # Get the folder to save the image in
        #     save_path = os.path.join(output_folder, letter_text)

        #     # if the output directory does not exist, create it
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     # write the letter image to a file
        #     count = counts.get(letter_text, 1)
        #     p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        #     cv2.imwrite(p, letter_image)

        #     # increment the count for the current key
        #     counts[letter_text] = count + 1

        for char_region, char in zip(char_regions, img_text):
            counts = crop_letter(char_region, char, gray, output_folder, counts)

def hard_captcha_preprocess(input_folder, output_folder):
    # Get a list of all the captcha images we need to process
    images = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    for (i, img_path) in enumerate(images):
        print(f"INFO: processing image {i+1}/{len(images)}")

        # Extract name of the file since it contains the captcha characters
        filename = os.path.basename(img_path)
        img_text = os.path.splitext(filename)[0]

        img, gray = complex_preprocess(img_path)

        char_regions = get_countours(img, lambda w, h: (((w / h) > 1.35) and (w > 22)) or (h > 28), num_chars = 10, cv2_chain = cv2.CHAIN_APPROX_NONE)

        if char_regions is None:
            continue

        # # find letters contours
        # contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours = contours[1] if imutils.is_cv3() else contours[0]

        # letter_image_regions =[]

        # # loop through contours and extract letter inside of each one
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area < 90:
        #         continue

        #     # get the rectangle that contains the contour
        #     (x, y, w, h) = cv2.boundingRect(contour)

        #     # When a contour is including more than one letter
        #     if (((w / h) > 1.35) and (w > 22)) or (h > 28):
        #         half_width =  int(w / 2)
        #         letter_image_regions.append((x, y, half_width, h))
        #         letter_image_regions.append((x + half_width, y, half_width, h))
        #     else:
        #         letter_image_regions.append((x, y, w, h))


        # # if we did not found 10 letters then letter extraction did not work and we skip it
        # if len(letter_image_regions) != 10:
        #     continue

        # # sort regions from left to right
        # letter_image_regions = sorted(letter_image_regions, key = lambda x: x[0])

        # save each letter on a separate image
        # for region, letter in zip(letter_image_regions, img_text):
        #     x, y, w, h = region
            
        #     # extract the letter from the original image with a 2 pixels margin around the edges
        #     letter_image  = img_gray[y-2 : y+h+2, x-2 : x+w+2]
            
        #     rect = cv2.rectangle(img_gray, (x-2, y-2), (x + w+2, y + h+2), (0, 255, 0), 2)

        #     # save letter
        #     save_path = os.path.join(output_folder, letter)

        #     # if the output directory does not exist, create it
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     # for numeration of images
        #     count = counts.get(letter, 1)

        #     p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        #     t = cv2.imwrite(p, letter_image)

        #     # update number of times each character was seen
        #     counts[letter] = count + 1

        for char_region, char in zip(char_regions, img_text):
            counts = crop_letter(char_region, char, gray, output_folder, counts)

def simple_preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray

def complex_preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # threshold the image with a costum mask because it is the best way to process this specific images
    lower = np.array([220,220,220])
    upper = np.array([255,255,255])
    my_mask = cv2.inRange(img, lower, upper)
    img = cv2.threshold(my_mask, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray


def get_countours(img, conjoined_condition, num_chars, cv2_chain):
    """
    Find contours in the image
    """
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2_chain)
    contours = contours[1] if imutils.is_cv3() else contours[0]

    char_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if (num_chars == 10) and  (area < 90):
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        if conjoined_condition(w, h):
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            char_regions.append((x, y, half_width, h))
            char_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            char_regions.append((x, y, w, h))

    if len(char_regions) != num_chars:
        return None

    char_regions = sorted(char_regions, key=lambda x: x[0])

    return char_regions


def crop_letter(char_region, char, img, out_folder, counts):
    """
    Crop the letter from the image
    """
    x, y, w, h = char_region
    
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

def captcha_kaggle_preprocess(input_folder, output_folder):
# Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print(f"INFO: processing image {i+1}/{len(captcha_image_files)}")

        # Extract name of the file since it contains the captcha characters
        filename = os.path.basename(captcha_image_file)
        captcha_characters = os.path.splitext(filename)[0]

        img = cv2.imread(os.path.join(input_folder, filename))

        # convert image to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # threshold the image with a costum mask because it is the best way to process this specific images
        lower = np.array([220,220,220])
        upper = np.array([255,255,255])
        my_mask = cv2.inRange(img, lower, upper)
        thresh = cv2.threshold(my_mask, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find letters contours
        img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = img_contours[1] if imutils.is_cv3() else img_contours[0]

        letter_image_regions =[]

        # loop through contours and extract letter inside of each one
        for contour in img_contours:
            area = cv2.contourArea(contour)
            if area < 90:
                continue

            # get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # When a contour is including more than one letter
            if (((w / h) > 1.35) and (w > 22)) or (h > 28):
                half_width =  int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))


        # if we did not found 10 letters then letter extraction did not work and we skip it
        if len(letter_image_regions) != 10:
            continue

        # sort regions from left to right
        letter_image_regions = sorted(letter_image_regions, key = lambda x: x[0])

        # save each letter on a separate image
        for region, letter in zip(letter_image_regions, captcha_characters):
            x, y, w, h = region
            
            # extract the letter from the original image with a 2 pixels margin around the edges
            letter_image  = img_gray[y-2 : y+h+2, x-2 : x+w+2]
            
            rect = cv2.rectangle(img_gray, (x-2, y-2), (x + w+2, y + h+2), (0, 255, 0), 2)

            # save letter
            save_path = os.path.join(output_folder, letter)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # for numeration of images
            count = counts.get(letter, 1)

            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            t = cv2.imwrite(p, letter_image)

            # update number of times each character was seen
            counts[letter] = count + 1
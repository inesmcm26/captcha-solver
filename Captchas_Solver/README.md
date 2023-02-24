<h1> Captcha Solver </h1>

Implementation of a solver for really simple captcha


This projects is divided into 2 parts:
1. Given a really simple captcha dataset with only digitis and upper case letters, the input is processed using OpenCV and then fed to a CNN.
2. Given a more complex captcha dataset, including also lower case digits, the input is processed with OpenCV and then fed into a new CNN.
3. The first dataset with the easiest captcha is fed into an updated version of the previous CNN used to solve the harder captcha. The network is extended using transfer learning: only the two last layers (the classification layers) are replaced by new ones and the model is re-trained with the feature extraction layers frozen.

<h3> Datasets

- simple_captcha: really simple captcha
- gard_captcha: similar to the previous one, but with lower case letters

<h3> Image processing </h3>

Given the simplicity of the captcha images, the processing is done by extracting each character from the image using some data treatment and OpenCV. Then, each character is used to train a CNN.

<h3> How to run it </h3>

A detailed explanation of each step can be found on the file [solver.ipynb](solver.ipynb)
In this file several steps are performed:
1. Data preprocessing: Doing this will create two new folders with the extacted digits from both simple and harder captcha;
2. Two models are trained. One using the simple data and the pther the harder images.
3. For each task, the results are visualized and evaluated.
4. Transfer learning: The model that trained on the harder data is used as the backbone of a new model that is also going to be used to solve simple captcha.
5. The transfer learning model and the simple model performances are compared.

<h3> Results </h3>
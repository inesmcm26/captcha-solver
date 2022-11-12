<h1> Captchas Solver </h1>

Implementation of a solver for really simple captchas


This projects is divided into 2 parts:
1. Given a really simple captchas dataset with only digitis and upper case letters, the input is processed using OpenCV and then fed to a CNN.
2. Given a more complex captchas dataset, including also lower case digits, the input is processed with OpenCV and then fed into a new CNN.
3. The first dataset with the easiest captcha is fed into an updated version of the previous CNN used to solve the harder captchas. The network is extended using transfer learning: only the two last layers (the classification layers) are replaced by new ones and the model is re-trained with the feature extraction layers frozen.

<h3> Datasets

- generated_captcha_images: really simple captcha
- kaggle-captchas: equal to the previous one, but with lower case letters

<h3> Image processing </h3>

Given the simplicity of the captcha images, the processing is done by extracting each digit the captcha and then train a CNN to identify each digit individualy

<h3> How to run it </h3>

1. Run the 'captcha_preprocessing.ipynb' file. Doing this will create two new folders with the extacted digits from both simple and harder captacha;
2. Run the 'captchas_solver.ipynb' file to see simple captchas being solved;
3. Run the 'harder_captchas_solver.ipynb' file to see a model solving harder captchas and then the same model updated using transfer learning solving the simple captchas.
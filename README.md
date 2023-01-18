# Machine Learning Literacy for Measurement Professionals: A Practical Tutorial

This is the demo for the paper.

## Data

All input data are saved in pickle format, in the 'data' folder. 
They are simulated mock data, used for demonstration purposes.
The results in the paper are from real data, not from the mock data presented here.

The whole dataset is in data.pkl.

The whole data are randomly split into train_valid/test set in 7/3 ratio, stratified by answerkey_id,
they are saved in data_train_valid.pkl and data_test.pkl.

The train_valid data set is further randomly split into training and validation sets in 7/3 ratio, stratified by answerky_id,
they are saved in data_train.pkl and data_valid.pkl.

All data have the following format: 

| column             | data type     | definition                                                                       |
|--------------------|-----------|----------------------------------------------------------------------------------|
| text               |  object   | raw candidate responses to the question, without any preprocessing     |
| answerkey_id       |  category | answer key id for each item              |


## Pipelines

All pipelines explained in the paper are in the 'automark' folder, they are described below.

### 1. Supervised non-NN pipelines

### (1) supervised_simple.py

A simple supervised learning scikit-learn pipeline, used as the base model.

### (2) supervised_search.py

A grid search/random search scikit-learn pipeline. The search space consists of various feature extraction methods and non-NN
classifiers.

### 2. Supervised NN pipelines

### (1) supervised_search_cnn_word.py

A simple word-level convolutional neural network model with random search of hyperparameters by the Keras Tuner. 

Note: we used the pre-trained GloVe word vectors for initialization of the embedding layer in this script. 
You have to download the GloVe file and put it into 'data' folder if you want to run the whole script successfully.
The GloVe file we used is glove.6B.50d.txt, it is downloaded 
and unzipped from http://nlp.stanford.edu/data/glove.6B.zip

### (2) supervised_search_cnn_char.py

The same model as above, except that inputs are changed from word to character n-grams, 
and there is no pre-trained embedding available (GloVe file is not needed in this script).

### (3) supervised_xlnet.py

Use the current state-of-the-art text classification model XLNet, add a dropout layer and a final classification layer on top of it.

### 3. Unsupervised pipeline

### (1) unsupervised_search.py

A grid search pipeline with customized classifier and pipeline classes, using KMeans clustering as the core model.

## How to Use

(1) Download the whole repository to your local machine, and open the project in a Python IDE.

(2) Python version 3.8 is required for this project, 
we recommend that you create a virtual environment with Python version 3.8 for this project.

(3) After activating the virtual environment, open a command window, 
make sure the current path is this project's root path,  
then use 'pip install -r requirements.txt' to install all the required packages.

Note: we used tensorflow-gpu to train the NN models, and it is listed in the requirements file. 
Tensorflow-gpu has some system requirements listed in 

https://www.nvidia.com/en-sg/data-center/gpu-accelerated-applications/tensorflow/#:~:text=System%20Requirements,8.0%20required%20for%20Pascal%20GPUs

If your system does not meet these requirements or you wish to use CPU to train NN models, 
you may have to make some adjustments when installing the packages.

(4) To try any pipeline listed in the 'Pipelines' section, 
simply run the corresponding script (as a whole or interactively) in a Python console.




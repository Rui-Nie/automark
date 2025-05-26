# Machine Learning Literacy for Measurement Professionals: A Practical Tutorial

<!-- Language & Domain -->
![Python](https://img.shields.io/badge/Python_3.8-3572A5?style=flat&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-6A5ACD?style=flat)<!-- Learning Types -->
![Supervised Learning](https://img.shields.io/badge/Supervised_Learning-1E90FF?style=flat)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-8A2BE2?style=flat)
![Semi-Supervised Learning](https://img.shields.io/badge/Semi--Supervised_Learning-7B68EE?style=flat)

<!-- Frameworks -->
![scikit-learn](https://img.shields.io/badge/Framework-scikit--learn-FFA500?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

## 1. Overview

<details open>

This repository contains the demo code accompanying the paper:  
**"Machine Learning Literacy for Measurement Professionals: A Practical Tutorial." -- Rui Nie, Qi Guo, Maxim Morin.**  

Explore practical applications of machine learning methods tailored for measurement professionals facing new challenges in digital assessment and automated scoring.

**Read the full paper here:** [https://doi.org/10.1111/emip.12539](https://doi.org/10.1111/emip.12539)

</details>

## 2. Problem Statement

<details open>

This demo showcases various machine learning methods designed to support the automated marking of short-answer questions in high-stakes exams. It is based on a real-world example from a national medical licensing examination, where candidates provide brief written responses to clinical case questions.

Currently, each response is independently scored by two human markers who use a predefined answer key. If the markers disagree, a senior "super marker" makes the final decision. Although reliable, this manual process is time-consuming, costly, and vulnerable to challenges such as fatigue, individual bias, and scoring inconsistencies over time.

This demo explores how machine learning models can assist or even replace one human marker, aiming to enhance both the efficiency and consistency of the marking process.

</details>

## 3. Model Summary
<details open>

This demo showcases a diverse set of machine learning techniques spanning classical models, deep learning architectures, and semi-supervised clustering methods.

### 3.1. Feature Engineering & Text Representation
We experimented with a wide range of input representations, including:
- Word unigrams and character n-grams (with varying n)
- Bag-of-Words (BoW) with and without TF-IDF weighting
- **GloVe word embeddings** (pretrained vectors, as an implementation of **transfer learning**)

### 3.2. Supervised Learning Models
**The supervised text classification models** include:
- **Classical ML**: Logistic Regression, Random Forest, Linear SVM, and **CatBoost** (a high-performance gradient boosting library optimized for categorical features)
- **Deep Learning**: 
  - **CNN** with 1D convolutional layers using multiple kernel sizes (1, 3, 5) to capture local semantic patterns
  - **Transfer learning** using **XLNet**, a state-of-the-art transformer-based language model at the time of research, implemented via Hugging Face Transformers

**Hyperparameter optimization** was conducted using:
- **Grid Search** for classical models and preprocessing pipeline tuning
- **Random Search** (via Keras Tuner) for CNN model architecture, exploring choices like:
  - Pooling strategies (global max pooling, concatenation, flattening)
  - Dense layer sizes
  - Learning rate and L2 regularization strength

### 3.3. Semi-Supervised Learning: KMeans with Anchor Labels
We designed a custom **KMeansClassifier** that enhances traditional clustering with minimal supervision. While KMeans is an unsupervised algorithm, our approach introduces a semi-supervised twist:

1. **Clustering**  
   Standard KMeans is applied to the feature vectors.

2. **Anchor Labeling**  
   For each cluster, we locate the sample closest to its centroid and assign it a true label (manually or from available data). This requires only **one labeled sample per cluster**.

3. **Label Propagation**  
   All other samples within the same cluster inherit the label of that central "anchor" sample.

4. **Evaluation**  
   Only propagated predictions (excluding anchor points) are used for evaluation, ensuring the supervision remains minimal. We report accuracy, F1-score, and visualize results via a confusion matrix.

This method allows us to generate interpretable clusters while using very limited labeled data—ideal for scenarios with high annotation cost.

To support flexible experimentation, we also implemented a **KMeansPipeline** class enabling preprocessing transformations (e.g., TF-IDF, dimensionality reduction) before clustering, and a **grid search** function to evaluate different preprocessing strategies end-to-end.

### 3.4. Evaluation
Model performance was primarily evaluated using **accuracy**. To deepen model understanding, we also examined:
- **Confusion Matrix**
- **ROC Curve** & **AUC**
- **F-score**
- **Cohen’s Kappa**
- **Feature importance** for interpretable classical models

Validation strategies:
- **5-fold cross-validation** for classical models
- **Stratified hold-out validation** (70/30) for deep learning models

</details>

## 4. Data Description
<details open>

All input data are stored in pickle format within the `data` folder. These are simulated mock datasets used solely for demonstration purposes.  
*Note: The results reported in the paper are based on real data, not the mock data provided here.*

The complete dataset is saved as `data.pkl`.

The data is randomly split into training/validation and test sets in a 70/30 ratio, stratified by `answerkey_id`. These subsets are saved as:
- `data_train_valid.pkl` (training + validation)
- `data_test.pkl` (test set)

The `data_train_valid.pkl` subset is further split into training and validation sets with the same 70/30 ratio, also stratified by `answerkey_id`. These are saved as:
- `data_train.pkl`
- `data_valid.pkl`

### Data Format

All datasets share the following structure:

| Column        | Data Type | Description                               |
|---------------|-----------|-------------------------------------------|
| `text`        | object    | Raw candidate responses to the question, without preprocessing |
| `answerkey_id`| category  | Answer key ID for each item               |

</details>

## 5. Pipeline Overview

<details open>

### 5.1. Supervised Learning: Non-Neural Network Pipelines
<details open>

These pipelines demonstrate traditional supervised learning workflows using the `scikit-learn` library. They assume access to labeled training data and are designed to classify candidate responses using hand-crafted textual features.

### 5.1.1. `supervised_simple.py`
<details open>

This is a baseline pipeline to establish a starting point for model performance. It uses word unigrams as input features and applies a random forest classifier with default hyperparameters. The focus is on simplicity and interpretability, making it a solid initial benchmark for further model development.

</details>

### 5.1.2. `supervised_search.py`

<details open>

This pipeline builds on the baseline by introducing **automated model selection** and **hyperparameter tuning** using both **grid search** and **random search** strategies. The search space explores:

- **Feature extraction methods**:  
  - Word unigrams (with and without stopword removal)  
  - Character n-grams (n = 2 to 5)

- **Counting and weighting schemes**:  
  - Raw counts  
  - **TF-IDF**

- **Classifier options**:  
  - **CatBoost**  
  - **Linear SVM**  
  - **Logistic Regression**  
  - **Random Forest**

This script represents a more advanced phase of the modeling process, demonstrating how different preprocessing techniques and model configurations are evaluated to improve performance through an iterative, data-driven approach.

</details>

</details>

### 5.2. Supervised Deep Learning (Neural Network) Pipelines

<details open>

These pipelines demonstrate deep learning approaches using the Keras API with TensorFlow backend. They explore both custom-built neural architectures and the application of state-of-the-art pretrained models.

To ensure efficient training and prevent overfitting, all models incorporate **early stopping** by monitoring validation loss and halting training when improvements plateau. This strategy helps maintain model generalizability across tasks.

#### 5.2.1. `supervised_search_cnn_word.py`

<details open>

This pipeline implements a **word-level Convolutional Neural Network (CNN)** for text classification, using **Keras Tuner’s random search** strategy for hyperparameter optimization. The model is a simple yet adaptable architecture designed to demonstrate how to construct and tune neural networks for natural language tasks.

The CNN architecture applies **parallel convolutional layers** with varying kernel sizes to capture local patterns in text sequences—like unigrams, bigrams, and trigrams—from word embeddings. This multi-scale approach enriches feature extraction without requiring a complex model.

One of the hyperparameters controls whether to include **Global Max Pooling applied directly on the embedding layer** alongside the pooled convolution outputs. This allows the model to flexibly combine global semantic information from the raw embeddings with localized features extracted by the convolutional layers, potentially improving representational richness.

A set of core hyperparameters are tuned automatically, including:

- Number of dense units (`dense_units`)
- Learning rate (`lr`)
- L2 regularization strength (`l2_value`)
- Pooling strategy on convolution output (`concat_pool`)
- Whether to include additional max pooling from the embedding layer (`max_pool`)

These tunable choices help the model adapt to various text patterns and dataset characteristics while keeping it lightweight and interpretable.

*This example is intended as a straightforward starting point for small- to medium-scale text classification tasks, emphasizing interpretability, modularity, and ease of experimentation.*

A key feature of this pipeline is its use of **transfer learning**: the embedding layer is initialized with pre-trained **GloVe** word vectors (`glove.6B.50d.txt`), providing a rich semantic foundation learned from large-scale text corpora.

> 📎 To run this script successfully, download and unzip the GloVe file from [nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip), and place `glove.6B.50d.txt` into the `data/` folder.

Below is a schematic diagram of the custom CNN architecture:

```
Input: Sequence of token IDs
 (shape = [batch_size, max_len])
        │
        ▼
Embedding Layer
 └─ Initialized with GloVe vectors (50D) or trainable embedding
        │
        ▼
╭────────────────────────────────────────────╮
│        Parallel Conv1D Layers              │
│  ┌──────────────────────────────────────┐  │
│  │ Conv1D (kernel size = 1)             │  │
│  │ Conv1D (kernel size = 3)             │  │
│  │ Conv1D (kernel size = 5)             │  │
│  │ → All use ReLU + L2 regularization   │  │
│  └──────────────────────────────────────┘  │
╰────────────────────────────────────────────╯
        │
        ▼
Concatenate Conv Outputs
        │
        ▼
Conditional Pooling (hyperparameter: `concat_pool`)
 ├─ GlobalMaxPooling1D() if True
 └─ Flatten() if False
        │
        ▼
Optional Concatenation
 └─ with GlobalMaxPooling1D(embedding) if `max_pool=True`
        │
        ▼
Dropout (0.1)
        │
        ▼
Dense Layer
 └─ Units = 16 / 32 / 64 (selected via hyperparameter)
    + L2 regularization
        │
        ▼
Dropout (0.1)
        │
        ▼
Output Layer
 └─ Dense + softmax (num_labels classes)

```
</details>

#### 5.2.2. `supervised_search_cnn_char.py`

<details open>

This script builds the **same CNN architecture** as described above but replaces word-level input with **character n-grams**. This variation is particularly useful for handling noisy or misspelled text, as character-based representations tend to be more resilient to surface-level variation.

Unlike the word-level version, this pipeline does **not use pre-trained embeddings**, since character-level embeddings are not commonly available. Instead, the embedding layer is trained from scratch.

</details>

#### 5.2.3. `supervised_xlnet.py`

<details open>

This pipeline fine-tunes **XLNet**, a pretrained state-of-the-art transformer, for text classification using the **Hugging Face Transformers library**. A **dropout layer** and a **classification head** are added on top of the base model to adapt it to our dataset.

This is an example of **transfer learning**: rather than training from scratch, the model starts with weights learned on large corpora and then adjusts them to fit the target task.

> **Tips**  
>  
> When using pretrained models like XLNet, you can **freeze** the base layers to save training time and resources, especially if your dataset is small or similar to the original training data.  
>  
> On the other hand, **fine-tuning** (unfreezing the layers) is useful when your data is domain-specific or significantly different—this helps the model learn new patterns relevant to your task.  
>  
> Not sure which to choose? Start with freezing the layers and see how it performs. If accuracy seems capped, try fine-tuning next!

In this script, we **free** the XLNet parameters so they can adapt more effectively to our data, allowing the model to learn domain-specific nuances while benefiting from a strong semantic foundation.

---

Together, these deep learning pipelines demonstrate two complementary strategies:

- Designing and tuning **custom architectures** for interpretability and modularity.
- Applying and adapting **advanced pretrained models** for powerful out-of-the-box performance.

Both approaches are valuable, depending on the goals, data size, and domain specificity of your task.

</details>

</details>


### 5.3. “Unsupervised” (Technically Semi-Supervised) Learning Pipeline

<details open>

### (1) unsupervised_search.py

This pipeline explores a clustering-based approach to augment human scoring—particularly useful for newly developed questions that lack labeled responses. While originally described as unsupervised in the published paper, the method is more accurately **semi-supervised**, since it relies on a small amount of human-labeled data to propagate labels across clusters.

The core idea is to apply **KMeans clustering** to group similar responses based on features like word or character n-grams. Instead of labeling every individual response, a human marker reviews just the **cluster representative** (the response closest to the cluster centroid), assigns it a label, and that label is propagated to all responses within the same cluster. This greatly reduces manual labeling effort.

Although the clustering algorithm operates without labels, the use of human-assigned labels at the cluster level introduces supervision—thus making this a **semi-supervised learning strategy** in practice.

The `KMeansPipeline` class supports chaining preprocessing steps before clustering. To identify the most effective configurations, a **grid search** is conducted across multiple dimensions, selecting the setup that yields the highest **label propagation accuracy** on validation data.

The search space includes:

- **Text vectorization options**:  
  - Word-level CountVectorizer  
  - Character-level n-grams (n = 2–4)

- **Weighting schemes**:  
  - Raw term counts  
  - **TF-IDF**  
  - Passthrough (no weighting)

- **Clustering configurations**:  
  - KMeans with varying numbers of clusters (e.g., 20, 50)

There is an inherent trade-off to consider: increasing the number of clusters can improve prediction accuracy, but it also raises the number of responses that require human review. Balancing this trade-off is key to achieving both efficient use of human labeling and high predictive accuracy.

This pipeline provides a cost-effective way to jump-start automated scoring when labeled data is scarce, by combining clustering with minimal human input.

</details>

## 6. User Guide

<details open>

**6.1. Download the repository** 

Download the repository and open the project folder in a Python IDE of your choice.

**6.2. Python 3.8 is required.**  

We recommend setting up a virtual environment using Python 3.8 to ensure compatibility.

**6.3. Install dependencies.**  

After activating the virtual environment, open a terminal or command prompt, navigate to the project root directory, and run: 

`pip install -r requirements.txt`


> **Note**: The `requirements.txt` includes `tensorflow-gpu`, which is used for training neural network models. GPU acceleration can significantly improve training performance, but it requires compatible hardware and drivers.  
>
> For system requirements, refer to:  
> [TensorFlow GPU - NVIDIA Requirements](https://www.nvidia.com/en-sg/data-center/gpu-accelerated-applications/tensorflow/#:~:text=System%20Requirements,8.0%20required%20for%20Pascal%20GPUs)
>
> If your system doesn’t meet these GPU requirements, or if you prefer to use the CPU version of TensorFlow, you may need to modify the dependency list accordingly.

**6.4. Run a pipeline.**  

To run any pipeline described in the **Pipeline Overview** section, execute the corresponding Python script—either all at once or interactively within your IDE. Each script can be explored step by step to better understand the workflow and output.

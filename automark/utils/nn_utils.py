import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Input, Embedding, Concatenate,\
                                    Conv1D, GlobalMaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
import keras_tuner as kt
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable

from automark.utils.label_encoder import LabelEncoder
from automark.utils.data_utils import preprocess_text, spacy_pipe


def build_char_vectorizer(data: pd.Series, ngram_max: int = 1) -> Tuple[TextVectorization, int, int]:
    """Build a character level vocabulary given the data.

    Args:
        data: the data to be used for building vocabulary.
        ngram_max: ngrams will be created up to this integer.

    Returns:
        A tuple of vectorizer, max_len, voc_size.
        The vectorizer is a vectorizer already adapted to the data.
        The max_len is the max length of the encoded vectors.
        The voc_size is the size of the built vocabulary.

    """
    # get max number of characters in text sentences
    max_len_char = int(data.apply(len).max())

    # calculate max length of encoded vectors
    max_len = int(max_len_char*ngram_max - ngram_max*(ngram_max-1)/2)

    vectorizer = TextVectorization(split='character', ngrams=ngram_max, output_sequence_length=max_len)
    vectorizer.adapt(data)
    voc_size = len(vectorizer.get_vocabulary())

    return vectorizer, max_len, voc_size


def build_word_vectorizer(data: pd.Series) -> Tuple[TextVectorization, dict, int]:
    """Build a word level vectorizer given the data.

    Args:
        data: the data to be used for building vocabulary.

    Returns:
        A tuple of vectorizer, word_index, max_len.
        The vectorizer is a vectorizer already adapted to the data.
        The word_index is a dict that maps each word to its index in the vocabulary.
        The max_len is the max length of the encoded vectors.

    """
    # get max number of words in text sentences
    max_len = int(data.str.split().apply(len).max())

    vectorizer = TextVectorization(output_sequence_length=max_len)
    vectorizer.adapt(data)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    return vectorizer, word_index, max_len


def build_embedding_matrix(embedding_path: str,
                           embedding_dim: int, word_index: dict) -> np.array:
    """Build an embedding matrix given the embedding file and word_index dict.

    Args:
        embedding_path: path to the file that stores the embedding.
        embedding_dim: dimension of the embedding vector.
        word_index: a dict that map each word to its index in the built vocabulary.

    Returns:
        the embedding matrix to be used as the initializer for the embedding layer.

    """
    # extract embeddings from file
    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in tqdm(f):
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, sep=' ')
            embeddings_index[word] = coefs

    # prepare embedding matrix
    num_tokens = len(word_index) + 2
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def encode_input(data: List[str], vectorizer: TextVectorization) -> np.array:
    """Encode list of text sentences to numeric vectors.

    Args:
        data: the text data to be encoded.

    Returns:
        The encoded numeric vectors.

    """
    return vectorizer(np.array([[s] for s in data])).numpy()


def nn_preprocess(data_dict: dict, encode_input_func: Callable) -> Tuple[dict, dict, int]:
    """ Preprocess data for neural networks.

    Args:
        data_dict: the data dict that holds all X and y.
        encode_input_func: the function that encode X.

    Returns:
        A tuple of X_processed, y_processed and num_labels.
        X_processed and y_processed are dicts that hold processed X and y.
        num_labels is the number of unique labels in y.

    """
    X_processed, y_processed = {}, {}

    label_encoder = LabelEncoder().fit(data_dict['y_train'])
    num_labels = len(label_encoder.encode_map)

    for setname in ('train', 'valid', 'test'):
        X_processed[setname] = spacy_pipe(data_dict[f'X_{setname}'], preprocess_text)
        X_processed[f'inputs_{setname}'] = encode_input_func(list(X_processed[setname]))
        y_processed[setname] = label_encoder.encode_label(data_dict[f'y_{setname}'])

    return X_processed, y_processed, num_labels


def build_cnn_model(hp: kt.HyperParameters,
                    num_labels: int,
                    max_len: int,
                    embedding_matrix: np.array = None,
                    embedding_input_dim: int = None,
                    embedding_output_dim: int = None) -> tf.keras.Model:
    """Define the keras model with hyperparameter choices.

    Args:
        hp: instance of kt.HyperParameters class.
        num_labels: number of unique labels in label data.
        max_len: max length of the input vectors.
        embedding_matrix: the embedding matrix to be initialized with, if exists.
        embedding_input_dim: input dimension of embedding layer,
                             if embedding_matrix is used, then no need to specify.
        embedding_output_dim: output dimension of embedding layer,
                             if embedding_matrix is used, then no need to specify.

    Returns:
        the keras model with hyperparameter choices left open.

    """
    # define hyperparameter choices
    concat_pool = hp.Boolean('concat_pool')
    max_pool = hp.Boolean('max_pool')
    dense_units = hp.Choice('dense_units', [16, 32, 64])
    lr = hp.Choice('lr', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    l2_value = hp.Choice('l2_value', [1e-5, 1e-4, 1e-3])

    # define inputs
    input_ids = tf.keras.Input(shape=(max_len,), name='input_ids', dtype='int32')

    # define embedding layer
    if embedding_matrix is not None:
        num_tokens, embedding_dim = embedding_matrix.shape
        embedding = Embedding(
            num_tokens, embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)
        )(input_ids)
    else:
        embedding = Embedding(embedding_input_dim, embedding_output_dim)(input_ids)

    # build rest of the model
    conv1 = Conv1D(32, 1, padding='same', activation='relu',
                   activity_regularizer=regularizers.l2(l2_value))(embedding)
    conv2 = Conv1D(32, 3, padding='same', activation='relu',
                   activity_regularizer=regularizers.l2(l2_value))(embedding)
    conv3 = Conv1D(32, 5, padding='same', activation='relu',
                   activity_regularizer=regularizers.l2(l2_value))(embedding)

    x = Concatenate()([conv1, conv2, conv3])

    if concat_pool:
        x = GlobalMaxPooling1D()(x)
    else:
        x = Flatten()(x)

    if max_pool:
        pool1 = GlobalMaxPooling1D()(embedding)
        x = Concatenate()([x, pool1])

    x = Dropout(0.1)(x)
    x = Dense(dense_units, activity_regularizer=regularizers.l2(l2_value))(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_labels, activation='softmax', activity_regularizer=regularizers.l2(l2_value))(x)

    # compile model
    model = tf.keras.Model(inputs=input_ids, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


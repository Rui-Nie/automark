from __future__ import annotations
import pandas as pd
import numpy as np
import tensorflow as tf


class LabelEncoder:
    """A class to encode series of label data to arrays of one-hot vectors and vice versa.

    Attributes:
        y: the label data to be used for generating encode_map and decode_map
        encode_map: the dict that can map all unique values in y into 0, 1, 2, ...
        decode_map: the reversed dict of encode_map

    """

    def __int__(self) -> None:
        self.y = None
        self.encode_map = None
        self.decode_map = None

    def fit(self, y: pd.Series) -> LabelEncoder:
        """Fit on label data to get encode_map and decode_map.

        Args:
            y: label data

        Returns:
            self

        """
        self.y = y
        y_unique = sorted(list(y.unique()))
        self.encode_map = dict(zip(y_unique, range(len(y_unique))))
        self.decode_map = dict(zip(range(len(y_unique)), y_unique))

        return self

    def encode_label(self, y: pd.Series) -> np.array:
        """Encode a series of data into an array of one-hot vectors.
           All elements in y must exist in encode_map.

        Args:
            y: the label data to be encoded into one-hot vectors.

        Returns:
            the encoded arrays of one-hot vectors.

        """
        y_new = np.array([self.encode_map[elem] for elem in y])
        y_cat = tf.keras.utils.to_categorical(y_new)

        return y_cat

    def decode_label(self, y_cat: np.array) -> pd.Series:
        """Decode the array of vectors to a list of elements as in original y.

        Args:
            y_cat: the array of vectors to be decoded.

        Returns:
            the decoded series of elements.

        """
        y_flat = y_cat.argmax(axis=1)
        y_new = pd.Series([self.decode_map[elem] for elem in y_flat])

        return y_new

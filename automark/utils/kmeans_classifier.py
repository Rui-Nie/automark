import pandas as pd
import numpy as np
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from numpy.typing import ArrayLike
from typing import List

from automark.utils.evaluation_utils import pred_evaluate


class KMeansClassifier:
    """A class that uses KMeans clustering model for clustering,
    then use the true  of the point closest to
    each clusters' center to predict label of all other points in that cluster.

    Attributes:
        km_clf: the instance of KMeans clustering model that will be used.
        center_index: a list of length n_clusters,
                      each index, value pair (i, v) indicates that for the cluster labelled i,
                      v is the index of the point in X that is closest to its center.
        y_pred: a dataframe, with column 'pred_label' storing predicted label for each corresponding point(row)
                in X, and column 'predicted' indicating whether the 'pred_label' in the same row is really
                a predicted label (True) or a provided true label (False). If value in 'predicted' is False,
                it means that its corresponding point's label is provided in center_labels, in other words, that
                point in X is closest to a cluster's center.
    """

    def __init__(self, km_model: KMeans) -> None:
        """
        Args:
            km_model: an instance of KMeans clustering model

        """
        self.km_clf = km_model
        self.center_index = None
        self.y_pred = None

    def fit(self, X: ArrayLike) -> List[int]:
        """Fit KMeans clustering on input data X,
        for each cluster, find one point(row) in X that is nearest to the center, save its index.
        Save a list of such indexes (one for each cluster) in attribute center_index, then return it.

        Args:
            X: {array-like, sparse matrix} of shape (n_sample, n_features),
               the same as the input X for KMeans().fit method

        Returns:
             the class attribute center_index.

        """
        self.km_clf.fit(X)
        self.center_index = pairwise_distances_argmin(self.km_clf.cluster_centers_, X).tolist()

        return self.center_index

    def predict(self, center_labels: list) -> pd.DataFrame:
        """Given input center_labels, with value v in index i indicating the true label
        of the point in X that is closest to cluster i's center, predict labels of all other points
        in cluster i as v. Save prediction result in attribute y_pred, and return it.

        Args:
            center_labels: a list of true labels, each index, value pair (i, v) indicates that
                           v is the true label of the point in X whose index is given by the ith value
                           of the class attribute center_index.
        Returns:
            the class attribute y_pred

        """

        if len(center_labels) != self.km_clf.n_clusters:
            raise ValueError('Input length must be the same as number of clusters.')

        self.y_pred = pd.DataFrame(
            {
                'pred_label': list(map(lambda x: center_labels[x], self.km_clf.labels_)),
                'predicted': [i not in set(self.center_index) for i in range(len(self.km_clf.labels_))]
            }
        )

        return self.y_pred

    def score(self, y_true: ArrayLike) -> float:
        """Given the array of true label y_true,
        calculate the mean accuracy of the previously predicted labels,
        return the mean accuracy.

        Args:
            y_true: the list of true labels, with the same length as number of samples in X.

        Returns:
            the mean accuracy score of the previously predicted labels.

        """
        if len(y_true) != self.y_pred.shape[0]:
            raise ValueError('Input length must be the same as number of samples.')

        return mean([int(y_true[i] == self.y_pred['pred_label'][i]) for i in range(len(y_true))
                     if i not in set(self.center_index)])

    def evaluate(self, y_true: ArrayLike) -> dict:
        """Given the array of true label y_true,
        return a dict of different evaluation metrics of the prediction,
        also display the plot of confusion matrix.

        Args:
            y_true: the list of true labels, with the same length as number of samples in X.

        Returns:
            a dict of evaluation metrics of the prediction, also plot the confusion matrix.

        """
        if len(y_true) != self.y_pred.shape[0]:
            raise ValueError('Input length must be the same as number of samples.')

        y_true_valid = np.array([y_true[i] for i in range(len(y_true)) if i not in set(self.center_index)])
        y_pred_valid = np.array([self.y_pred['pred_label'][i] for i in range(len(y_true))
                                 if i not in set(self.center_index)])

        return pred_evaluate(y_pred=y_pred_valid, y_true=y_true_valid)

    def fit_predict(self, X: ArrayLike, y_true: ArrayLike) -> dict:
        """If true labels are available for all data already, automatically
        conduct KMeans clustering on input X, then for each cluster, use the
        true label of the point in X that is closest to its center to predict
        labels of all other points in that cluster, return a dictionary with both the prediction dataframe
        y_pred and its prediction accuracy score.

        Args:
            X: {array-like, sparse matrix} of shape (n_sample, n_features),
               the same as the input X for KMeans().fit method.

            y_true: array-like, with length = n_sample, the true labels of the samples in X.

        Returns:
            a dictionary, 'y_pred' stores the prediction dataframe, which is the class attribute y_pred,
            'pred_accuracy' is the prediction accuracy.

        """
        self.fit(X)
        center_labels = [y_true[index] for index in self.center_index]
        self.predict(center_labels)

        return {
            'pred_accuracy': self.score(y_true),
            'y_pred': self.y_pred
        }


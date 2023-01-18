import pandas as pd
from sklearn.cluster import KMeans
from numpy.typing import ArrayLike
from typing import Sequence, List
from tqdm import tqdm

from automark.utils.kmeans_classifier import KMeansClassifier


class KMeansPipeline(KMeansClassifier):
    """A class that enables piping of transformers with the customized KMeansClassifier model.

    Attributes:
        pipe: the pipeline that will be used, in the form of a list of model instances.

    """

    def __init__(self, pipe: Sequence) -> None:
        if len(pipe) == 0 or type(pipe[-1]) != KMeans:
            raise ValueError('The last element in the pipe must be a KMeans instance.')
        self.pipe = pipe
        super().__init__(pipe[-1])

    def _transform(self, X: ArrayLike) -> ArrayLike:
        """Apply the fit_transform of all models before the final KMeans model to the data,
        in the order they appear in the list.

        Returns:
            the transformed data.

        """
        X_new = X.copy()

        if len(self.pipe) > 1:
            for i in range(len(self.pipe)-1):
                if self.pipe[i] != 'passthrough':
                    X_new = self.pipe[i].fit_transform(X_new)

        return X_new

    def fit(self, X: ArrayLike) -> List[int]:
        """Fit the data using KMeansClassifier after it is transformed.

        Args:
            X: {array-like, sparse matrix} of shape (n_sample, n_features),
               the same as the input X for KMeans().fit method.

        Returns:
            the center_index list.

        """
        X_new = self._transform(X)

        return super().fit(X_new)


def kmeans_gridsearch(pipe_list: List[Sequence], X: ArrayLike, y_true: ArrayLike) -> pd.DataFrame:
    """Loop through the pipelines provided in the pipe_list, provide prediction accuracy for all of them.

    Args:
        pipe_list: list of pipelines, each pipeline is represented in the form of a sequence of model instances.
                   The last model in each pipeline must be a KMeans instance.
        X: {array-like, sparse matrix} of shape (n_sample, n_features),
           the same as the input X for KMeans().fit method
        y_true: array-like, with length = n_sample, the true labels of the samples in X.

    Returns:
        The result dataframe, the column 'pipe' records the pipeline settings,
        the column 'pred_accuracy' records the prediction accuracy of the pipeline.
        The dataframe is sorted in descending order of 'pred_accuracy'.

    """
    result_df = pd.DataFrame({'pipe': pipe_list})
    tqdm.pandas()
    result_df['pred_accuracy'] \
        = result_df.progress_apply(lambda row: KMeansPipeline(row['pipe']).fit_predict(X, y_true)['pred_accuracy'],
                                   axis=1)
    result_df = result_df.sort_values(by=['pred_accuracy'], ascending=False).reset_index(drop=True)

    return result_df





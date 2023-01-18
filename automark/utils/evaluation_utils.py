import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, f1_score, \
                            confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

from automark.utils.label_encoder import LabelEncoder


def plot_roc_curve(y_pred_proba: ArrayLike, y_true: ArrayLike) -> None:
    """Plot a macro-average ROC curve, from predicted label probabilities and true labels.

    Args:
        y_pred_proba: predicted probability of each class, of shape (n_sample, n_features),
                      it can be None if doesn't exist.
        y_true: true labels of shape (n_sample, 1).

    """
    y_true_proba = label_binarize(y_true, classes=sorted(y_true.unique()))
    n_classes = y_true_proba.shape[1]

    fpr, tpr = {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_proba[:, i], y_pred_proba[:, i])

    # aggregate all false positive rate
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # interpolate all ROC curves at each point and compute the average
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(
        fpr['macro'],
        tpr['macro'],
        color='black',
        linestyle='solid'
    )

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def pred_evaluate(y_pred: ArrayLike, y_true: ArrayLike, y_pred_proba: ArrayLike = None) -> dict:
    """Return several evaluation metrics, and plot a confusion matrix and a ROC curve plot,
       from predicted and true labels.

    Args:
        y_pred: predicted labels of shape (n_sample, 1)
        y_pred_proba: predicted probability of each class, of shape (n_sample, n_features),
                      it can be None if doesn't exist.
        y_true: true labels of shape (n_sample, 1)

    Returns:
        a dictionary with several evaluation metrics
        Also automatically displays a confusion matrix plot and a ROC curve plot.
    """
    y_unique_labels = np.unique(y_true)

    evaluation_dict = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=y_unique_labels)
    }

    if y_pred_proba is not None:
        evaluation_dict['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        # plot macro-average roc curve
        plot_roc_curve(y_pred_proba, y_true)

    # plot confusion matrix
    ConfusionMatrixDisplay(evaluation_dict['confusion_matrix'], display_labels=y_unique_labels).plot()

    return evaluation_dict


def clf_evaluate(estimator, X: ArrayLike, y: ArrayLike) -> dict:
    """Return several evaluation metrics, plot a confusion matrix and a ROC curve plot,
       from a sklearn estimator or pipeline with estimator at the end, input data and true labels.

    Args:
        estimator: a sklearn estimator or pipeline object that ends with an estimator
        X: {array-like, sparse matrix} of shape (n_sample, n_features)
        y: true labels of the input data, of shape (n_sample, 1)

    Returns:
        a dictionary with several evaluation metrics
    """
    y_pred = estimator.predict(X)
    y_pred_proba = estimator.predict_proba(X)

    return pred_evaluate(y_pred=y_pred, y_true=y, y_pred_proba=y_pred_proba)


def nn_evaluate(model, X: ArrayLike, y: ArrayLike) -> dict:
    """Return several evaluation metrics, plot a confusion matrix and a ROC curve plot,
       from a keras model, input data and true labels.

    Args:
        model: a keras model
        X: {array-like, sparse matrix} of shape (n_sample, n_features)
        y: true labels of the input data, of shape (n_sample, 1)

    Returns:
        a dictionary with several evaluation metrics
    """
    y_pred_proba = model.predict(X)
    label_encoder = LabelEncoder().fit(y)
    y_pred = label_encoder.decode_label(y_pred_proba)

    return pred_evaluate(y_pred=y_pred, y_true=y, y_pred_proba=y_pred_proba)


def get_most_important_features(pipeline: Pipeline, n: int) -> Union[list, dict]:
    """Return top n important features with their feature importance from a sklearn pipeline.

    Args:
        pipeline: a sklearn pipeline, the first step must be a vectorizer that has vocabulary_ attribute,
                  the last step must be one of the 3 classifiers: SGDClassifier, RandomForestClassifier,
                  CatBoostClassifier.
        n: number of most important features you want to get

    Return:
        If the pipeline has an SGDClassifier, it will return a dict with class label as key, and a list of
        top n important features with corresponding coefficients as feature importance scores;
        if the pipeline has the other 2 classifiers, it will return a list of top n important features
        with corresponding feature importance scores defined by each classifier.

    """
    vocab = pipeline.steps[0][1].vocabulary_
    index_token_dict = {v: k for k, v in vocab.items()}

    clf = pipeline.steps[-1][1]

    if type(clf) == SGDClassifier:
        # use absolute value of coef as feature importance indicator for SGDClassifier
        # need to do it for each class
        importances = clf.coef_
        top_n_index = (-abs(importances)).argsort(axis=-1)[:, :n]
        top_n_feature = {}
        for i in range(len(clf.classes_)):
            top_n_feature[clf.classes_[i]] = [(index_token_dict[ind], importances[i, ind]) for ind in top_n_index[i, :]]
    else:
        if type(clf) == RandomForestClassifier:
            importances = clf.feature_importances_
        elif type(clf) == CatBoostClassifier:
            importances = clf.get_feature_importance()
        top_n_index = (-importances).argsort()[:n]
        top_n_feature = [(index_token_dict[ind], importances[ind]) for ind in top_n_index]

    return top_n_feature


import os
import re
import string
from pprint import pprint
import pandas as pd
import numpy as np
from typing import Iterable, Callable, Dict
import spacy


def preprocess_text(text: str) -> str:
    """Preprocessing of the text in the sequence below:
    (1) Turn all letters to lowercase
    (2) Replace all punctuations with space
    (3) Turn multiple consecutive spaces into one
    (4) Strip text

    Args:
        text: input text

    Returns:
        the preprocessed text

    """
    text = re.sub(f'[{string.punctuation}]+', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def spacy_pipe(texts: Iterable[str], preprocess_func: Callable[[str], str], batch_size: int = 128) -> pd.Series:
    """Use spacy pipe to batch process a series of text.

    Args:
        texts: a sequence of text strings
        preprocess_func: a customized function to add more preprocessing to the text tokens
        batch_size: batch size for spacy pipe

    Returns:
        a series of processed text strings

    """
    # 'en_core_web_sm' below is downloaded by executing the following in cmd:
    # python -m spacy download en_core_web_sm
    spacy_nlp = spacy.load('en_core_web_sm')

    clean_texts = []
    for doc in spacy_nlp.pipe(texts, batch_size=batch_size):
        lemma_text = ' '.join([str(token.lemma_) for token in doc])
        clean_text = preprocess_func(lemma_text)
        clean_texts.append(clean_text)

    return pd.Series(clean_texts)


def retrieve_data(data_path: str) -> Dict[str, pd.Series]:
    """Retrieve needed training and testing data from pickled dfs.

    Args:
        data_path: the path where pickled dfs are stored.

    Returns:
        A dict of data series, the keys are their names.

    """
    data_dict = dict()

    for gp_name in ('train_valid', 'train', 'valid', 'test'):
        data = pd.read_pickle(os.path.join(data_path, f'data_{gp_name}.pkl')).reset_index(drop=True)
        data_dict[f'X_{gp_name}'], data_dict[f'y_{gp_name}'] = data['text'], data['answerkey_id']

    return data_dict


def print_cv_result(cv_result: dict) -> None:
    """Print out cross validation result given the result dict,
    also print out mean training and test score.

    Args:
        cv_result: the cross validation result dict

    """
    print('Cross validation result:')
    pprint(cv_result)
    for key in ('train_score', 'test_score'):
        print(f'Cross validation mean {key}:', np.mean(cv_result[key]))

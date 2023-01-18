import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from automark.utils.data_utils import preprocess_text, spacy_pipe, retrieve_data
from automark.utils.evaluation_utils import clf_evaluate, get_most_important_features

# retrieve data
root_path = r'the\project\root\path'
data_path = os.path.join(root_path, 'data')
data_dict = retrieve_data(data_path)
X_train, y_train, X_test, y_test = (data_dict[key] for key in ('X_train_valid', 'y_train_valid', 'X_test', 'y_test'))

# data preprocessing
X_train, X_test = (spacy_pipe(data, preprocess_text) for data in (X_train, X_test))

# catboost parameters
catboost_param = {
    'eval_metric': 'Accuracy',
    'loss_function': 'MultiClass',
    'logging_level': 'Silent'
}

# common elements in param_grid
common_param_grid = {
    'tfidf': [TfidfTransformer(), 'passthrough'],
    'clf': [
        CatBoostClassifier(**catboost_param),
        SGDClassifier(loss='hinge'),
        SGDClassifier(loss='log'),
        RandomForestClassifier()
    ]
}

# different elements in param_grid
diff_param_grid = [
    {
        'vect__analyzer': ['word'],
        'vect__stop_words': [None, 'english'],
    },
    {
        'vect__analyzer': ['char'],
        'vect__ngram_range': [(i, j) for i in range(2, 5) for j in range(i, 5)],
    },
]

# combine the common and different elements to get the full param_grid
param_grid = [{**elem, **common_param_grid} for elem in diff_param_grid]

pipeline = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ]
)

# Choose one of the search methods below
# 1. exhaustive grid search
# search = GridSearchCV(pipeline,
#                       param_grid=param_grid,
#                       cv=5,
#                       n_jobs=6,
#                       return_train_score=True,
#                       verbose=2)

# 2. random search with number of trials = n_iter
search = RandomizedSearchCV(pipeline,
                            param_distributions=param_grid,
                            cv=5,
                            n_jobs=6,
                            return_train_score=True,
                            n_iter=10,
                            verbose=2)

search.fit(X_train, y_train)
result_df = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'])

# examine the result_df to check for overfitting/underfitting
# if not satisfied, experiment with other hyperparameter or model choices
# if satisfied with the best estimator, test it on the test set
best_pipeline = search.best_estimator_
evaluation_dict = clf_evaluate(best_pipeline, X_test, y_test)
most_important_features = get_most_important_features(best_pipeline, 10)

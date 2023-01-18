import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from automark.utils.data_utils import preprocess_text, spacy_pipe, retrieve_data, print_cv_result
from automark.utils.evaluation_utils import clf_evaluate, get_most_important_features

# retrieve data
root_path = r'the\project\root\path'
data_path = os.path.join(root_path, 'data')
data_dict = retrieve_data(data_path)
X_train, y_train, X_test, y_test = (data_dict[key] for key in ('X_train_valid', 'y_train_valid', 'X_test', 'y_test'))

# data preprocessing
X_train, X_test = (spacy_pipe(data, preprocess_text) for data in (X_train, X_test))

# a simple pipeline
pipeline = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('clf', RandomForestClassifier())
    ]
)

# cross validation
cv_results = cross_validate(estimator=pipeline,
                            X=X_train,
                            y=y_train,
                            scoring='accuracy',
                            return_train_score=True,
                            verbose=2)


print_cv_result(cv_results)

# examine the printed result to check for overfitting/underfitting
# if not satisfied, experiment with other hyperparameter or model choices
# if satisfied, train on the whole training set and test on test set
pipeline.fit(X_train, y_train)
evaluation_dict = clf_evaluate(pipeline, X_test, y_test)
most_important_features = get_most_important_features(pipeline, 10)





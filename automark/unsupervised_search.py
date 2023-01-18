import os
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans

from automark.utils.data_utils import preprocess_text, spacy_pipe, retrieve_data
from automark.utils.kmeans_pipeline import KMeansPipeline, kmeans_gridsearch

# retrieve data
root_path = r'the\project\root\path'
data_path = os.path.join(root_path, 'data')
data_dict = retrieve_data(data_path)
X_train, y_train, X_test, y_test = (data_dict[key] for key in ('X_train_valid', 'y_train_valid', 'X_test', 'y_test'))

# data preprocessing
X_train, X_test = (spacy_pipe(data, preprocess_text) for data in (X_train, X_test))

# prepare pipelines
model_by_step = [
    [
        CountVectorizer(),
        CountVectorizer(analyzer='char', ngram_range=(2, 3)),
        CountVectorizer(analyzer='char', ngram_range=(3, 3)),
        CountVectorizer(analyzer='char', ngram_range=(2, 4))
    ],
    [
        TfidfTransformer(),
        'passthrough'
    ],
    [
        KMeans(n_clusters=20),
        KMeans(n_clusters=50)
    ]
]

pipe_list = list(itertools.product(*model_by_step))

# grid search on training set
result_df = kmeans_gridsearch(pipe_list, X_train, y_train)
result_df.to_csv(os.path.join(root_path, 'outputs/unsupervised_search_result_df.csv'))

# select the best pipeline and test on test data
# pay attention to number of clusters vs number of samples
best_pipe_structure = result_df.loc[0, 'pipe']
best_pipeline = KMeansPipeline(best_pipe_structure)
pred_result = best_pipeline.fit_predict(X_test, y_test)
evaluation_dict = best_pipeline.evaluate(y_test)

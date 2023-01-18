import os
from functools import partial
import tensorflow as tf
import keras_tuner as kt

from automark.utils.data_utils import retrieve_data
from automark.utils.nn_utils import nn_preprocess, build_word_vectorizer, encode_input, \
                                    build_embedding_matrix, build_cnn_model
from automark.utils.evaluation_utils import nn_evaluate

# retrieve data
root_path = r'the\project\root\path'
data_path = os.path.join(root_path, 'data')
data_dict = retrieve_data(data_path)
y_test = data_dict['y_test']

# preprocess data
vectorizer, word_index, max_len = build_word_vectorizer(data_dict['X_train'])
embedding_matrix = build_embedding_matrix(embedding_path=os.path.join(data_path, 'glove.6B.50d.txt'),
                                          embedding_dim=50, word_index=word_index)

encode_input_vec = partial(encode_input, vectorizer=vectorizer)

X_processed, y_processed, num_labels = nn_preprocess(data_dict, encode_input_vec)

# random search
model_partial = partial(build_cnn_model, num_labels=num_labels, max_len=max_len,
                        embedding_matrix=embedding_matrix)

tuner = kt.RandomSearch(
    hypermodel=model_partial,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    overwrite=True,
    directory=os.path.join(root_path, 'outputs/cnn'),
    project_name='word_search',
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(os.path.join(root_path, 'outputs/cnn/word_tensorboard'))
]

tuner.search(
    X_processed['inputs_train'],
    y_processed['train'],
    batch_size=128,
    verbose=1,
    validation_data=(X_processed['inputs_valid'], y_processed['valid']),
    shuffle=True,
    callbacks=callbacks,
    epochs=2000
)

# evaluate best model on test set
best_model = tuner.get_best_models()[0]
best_model.summary()
evaluation_dict = nn_evaluate(best_model, X_processed['inputs_test'], y_test)

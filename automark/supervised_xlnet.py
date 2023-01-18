import os
from transformers import TFXLNetModel, XLNetTokenizer
import tensorflow as tf
import numpy as np
from typing import List, Tuple
from functools import partial

from automark.utils.data_utils import retrieve_data
from automark.utils.nn_utils import nn_preprocess
from automark.utils.evaluation_utils import nn_evaluate


def encode_input_xlnet(data: List[str], xlnet_model: str = 'xlnet-base-cased', max_len: int = 32) \
                       -> Tuple[np.array, np.array]:
    """Encode data to inputs for xlnet model.

    Args:
        data: the data to be encoded.
        xlnet_model: the name of the xlnet to be used.
        max_len: the max length of the encoded vectors.

    Returns:
        A tuple of input_ids and attention_masks, to be used as inputs for xlnet.

    """
    tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
    tokenized_data = [tokenizer.encode_plus(sent, max_length=max_len, padding='max_length',
                                            add_special_tokens=True, truncation=True) for sent in data]
    input_ids = np.asarray([d['input_ids'] for d in tokenized_data])
    attention_masks = np.asarray([d['attention_mask'] for d in tokenized_data])

    return input_ids, attention_masks


def build_model(num_labels: int, xlnet_model: str = 'xlnet-base-cased',
                freeze_xlnet: bool = False, max_len: int = 32) -> tf.keras.Model:
    """Define the keras model to be trained.

    Args:
        num_labels: number of unique labels in label data.
        xlnet_model: the name of the xlnet model to be used.
        freeze_xlnet: whether to freeze the xlnet model layers or not.
        max_len: max length of the input vectors.

    Returns:
        the keras model.
    """
    # define inputs
    input_ids = tf.keras.Input(shape=(max_len,), name='input_ids', dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), name='attention_masks', dtype='int32')

    # call xlnet model
    xlnet = TFXLNetModel.from_pretrained(xlnet_model)
    xlnet.trainable = not freeze_xlnet
    xlnet_encodings = xlnet([input_ids, attention_masks])[0]
    cls_encodings = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)

    # add a dropout layer and a classification head
    cls_encodings = tf.keras.layers.Dropout(0.1)(cls_encodings)
    outputs = tf.keras.layers.Dense(num_labels, activation='softmax', name='outputs')(cls_encodings)

    # compile model
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# retrieve data
root_path = r'the\project\root\path'
data_path = os.path.join(root_path, 'data')
data_dict = retrieve_data(data_path)
y_test = data_dict['y_test']

# preprocess data
max_len = 32
xlnet_model_name = 'xlnet-base-cased'
encode_input_xlnet_base = partial(encode_input_xlnet, xlnet_model=xlnet_model_name, max_len=max_len)

X_processed, y_processed, num_labels = nn_preprocess(data_dict, encode_input_xlnet_base)

# train model
model = build_model(num_labels=num_labels, xlnet_model=xlnet_model_name,
                    freeze_xlnet=False, max_len=max_len)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(os.path.join(root_path, 'outputs/xlnet_tensorboard'))
]

history = model.fit(X_processed['inputs_train'],
                    y_processed['train'],
                    batch_size=64,
                    verbose=2,
                    validation_data=(X_processed['inputs_valid'], y_processed['valid']),
                    shuffle=True,
                    callbacks=callbacks,
                    epochs=1000)

# evaluate model on test set
evaluation_dict = nn_evaluate(model, X_processed['inputs_test'], y_test)






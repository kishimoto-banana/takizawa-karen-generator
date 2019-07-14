import pickle
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_EPOCHS = 60
FILE_PATH = 'data/corpus_test.pkl'
MODEL_CHECKPOINT_PATH = 'model/word-wise-karen_{epoch:02d}.h5'
# MODEL_FILE_PATH = 'model/tying_word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/multilayer_word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/tying_multilayer_word-wise-karen.h5'
MODEL_FILE_PATH = 'model/length_sequential_tying_word-wise-karen.h5'
GENERATED_TEXT_PATH = 'generated_text/word-wise-karen_diversity_{}.txt'

DELIMITER = '+++$+++'
SEQ_MAX_LEN = 5
STEP = 1


def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.backend.mean(
        tf.keras.backend.categorical_crossentropy(
            tf.convert_to_tensor(y_pred), tf.convert_to_tensor(y_true)),
        axis=-1)
    perplexity = tf.keras.backend.exp(cross_entropy)
    return perplexity


def create_model(vocab_size, embedding_dim=128, hidden=128):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='embedding',
            input_length=SEQ_MAX_LEN))
    model.add(tf.keras.layers.LSTM(hidden))
    # for sequential
    # model.add(tf.keras.layers.LSTM(hidden, return_sequences=True))
    model.add(
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(
            x,
            tf.keras.backend.transpose(
                model.get_layer('embedding').embeddings))))
    model.add(tf.keras.layers.Activation('softmax'))

    return model


# def create_model_multilayer(vocab_size, embedding_dim=128, hidden=128, dropout_rate=0.5):
#     model = tf.keras.models.Sequential()
#     model.add(
#         tf.keras.layers.Embedding(
#             input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='embedding'))
#     model.add(tf.keras.layers.Dropout(dropout_rate))
#     model.add(tf.keras.layers.LSTM(hidden, return_sequences=True, dropout=dropout_rate))
#     model.add(tf.keras.layers.LSTM(hidden, dropout=dropout_rate))
#     model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(x, tf.keras.backend.transpose(model.get_layer('embedding').embeddings))))
#     model.add(tf.keras.layers.Activation('softmax'))
#
#     return model

# load corpus
with open(FILE_PATH, 'rb') as f:
    corpus = pickle.load(f)
    word_to_id = pickle.load(f)
    id_to_word = pickle.load(f)

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_words = []
for i in range(0, len(corpus) - SEQ_MAX_LEN, STEP):
    sentences.append(corpus[i:i + SEQ_MAX_LEN])
    next_words.append(corpus[i + SEQ_MAX_LEN])

print('nb sequences:', len(sentences))

print('Vectorization...')
vocab_size = len(word_to_id)
x = np.array(sentences)
y = np.array(next_words)
y_true = tf.keras.utils.to_categorical(y, vocab_size)

# model = tf.keras.models.load_model(MODEL_FILE_PATH)

model = create_model(vocab_size)
model.load_weights(MODEL_FILE_PATH)

y_pred = model.predict(x)

# for sequential
# y_pred = y_pred[:, -1, :]

print(perplexity(y_true, y_pred))

import pickle
import sys
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_EPOCHS = 60
FILE_PATH = 'data/corpus.pkl'
MODEL_CHECKPOINT_PATH = 'model/word-wise-karen_{epoch:02d}.h5'
# MODEL_FILE_PATH = 'model/word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/multilayer_word-wise-karen.h5'
MODEL_FILE_PATH = 'model/tying_word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/tying_multilayer_word-wise-karen.h5'
# MODEL_FILE_PATH = 'model/length_sequential_tying_word-wise-karen.h5'
GENERATED_TEXT_PATH = 'generated_text/word-wise-karen_diversity_{}.txt'

DELIMITER = '+++$+++'
SEQ_MAX_LEN = 5
STEP = 1
GENERATE_MAX_WORDS = 300

diversities = [0.5, 1.0]


def create_model(vocab_size, embedding_dim=128, hidden=128):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='embedding'))

    # for sequential
    # model.add(
    #     tf.keras.layers.Embedding(
    #         input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='embedding', input_length=SEQ_MAX_LEN))
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


def create_model_multilayer(vocab_size,
                            embedding_dim=128,
                            hidden=128,
                            dropout_rate=0.5):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='embedding'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(
        tf.keras.layers.LSTM(
            hidden, return_sequences=True, dropout=dropout_rate))
    model.add(tf.keras.layers.LSTM(hidden, dropout=dropout_rate))
    model.add(
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(
            x,
            tf.keras.backend.transpose(
                model.get_layer('embedding').embeddings))))
    model.add(tf.keras.layers.Activation('softmax'))

    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, start_index=1257):

    # ”みなさん”から始めたい
    for diversity in diversities:
        print('----- diversity:', diversity)

        generated = ''
        sentence = corpus[start_index:start_index + SEQ_MAX_LEN]
        sentence_word = [id_to_word[word_id] for word_id in sentence]
        generated += ''.join(sentence_word)
        print('----- Generating with seed: "' + ''.join(sentence_word) + '"')
        # sys.stdout.write(generated)

        for i in range(GENERATE_MAX_WORDS):
            x_pred = np.array(sentence).reshape((1, -1))

            preds = model.predict(x_pred, verbose=0)[0]
            # for sequential
            # preds = preds[-1, :]

            next_index = sample(preds, diversity)
            next_word = id_to_word[next_index]

            if next_word == '<eos>':
                break

            generated += next_word
            sentence = sentence[1:]
            sentence.append(next_index)

        generated = generated.replace('N:N', '9:00')
        generated = generated.replace('N/N', '8/10')
        generated = generated.replace('N', '5')
        print(generated)


# load corpus
with open(FILE_PATH, 'rb') as f:
    corpus = pickle.load(f)
    word_to_id = pickle.load(f)
    id_to_word = pickle.load(f)

vocab_size = len(word_to_id)

# model = tf.keras.models.load_model(MODEL_FILE_PATH)

model = create_model(vocab_size)
model.load_weights(MODEL_FILE_PATH)

for i in range(3):
    generate(model)

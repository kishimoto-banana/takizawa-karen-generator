import pickle
import sys
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_EPOCHS = 100
FILE_PATH = 'data/corpus.pkl'
MODEL_CHECKPOINT_PATH = 'model/tying_multilayer_word-wise-karen_{epoch:02d}.h5'
MODEL_FILE_PATH = 'model/tying_multilayer_word-wise-karen.h5'
GENERATED_TEXT_PATH = 'generated_text/tying_multilayer_word-wise-karen_diversity_{}.txt'

DELIMITER = '+++$+++'
SEQ_MAX_LEN = 5
STEP = 1
GENERATE_MAX_WORDS = 300

diversities = [0.2, 0.5, 1.0]


def create_model(vocab_size, embedding_dim=128, hidden=128, dropout_rate=0.5):
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


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    # ”みなさん”から始めたい
    start_index = 1257
    for diversity in diversities:
        print('----- diversity:', diversity)

        generated = ''
        sentence = corpus[start_index:start_index + SEQ_MAX_LEN]
        sentence_word = [id_to_word[word_id] for word_id in sentence]
        generated += ''.join(sentence_word)
        print('----- Generating with seed: "' + ''.join(sentence_word) + '"')
        sys.stdout.write(generated)

        for i in range(GENERATE_MAX_WORDS):
            x_pred = np.array(sentence).reshape((1, -1))

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = id_to_word[next_index]

            if next_word == '<eos>':
                break

            generated += next_word
            sentence = sentence[1:]
            sentence.append(next_index)

            sys.stdout.write(next_word)
            sys.stdout.flush()

        with open(GENERATED_TEXT_PATH.format(diversity), mode='a') as f:
            f.write(f'epoch: {epoch}')
            f.write('\n')
            f.write(generated)
            f.write('\n')
            f.write(DELIMITER)
            f.write('\n')

        print()


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

print('Build model...')
model = create_model(vocab_size)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_CHECKPOINT_PATH, save_weights_only=True)

model.fit(
    x,
    y,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    callbacks=[print_callback, checkpoint_callback])

model.save(MODEL_FILE_PATH)

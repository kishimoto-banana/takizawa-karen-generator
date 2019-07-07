import io
import sys
import random
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_EPOCHS = 60
FILE_PATH = 'data/karen_insta.txt'
MODEL_CHECKPOINT_PATH = 'model/char-wise-karen_{epoch:02d}.h5'
MODEL_FILE_PATH = 'model/char-wise-karen.h5'
GENERATED_TEXT_PATH = 'generated_text/char-wise-karen_diversity_{}.txt'

DELIMITER = '+++$+++'
SEQ_MAX_LEN = 8
STEP = 1
GENERATE_CHARS = 400

diversities = [0.2, 0.5, 1.0]


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

    start_index = random.randint(0, len(text) - SEQ_MAX_LEN - 1)
    for diversity in diversities:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index:start_index + SEQ_MAX_LEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(GENERATE_CHARS):
            x_pred = np.zeros((1, SEQ_MAX_LEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

        with open(GENERATED_TEXT_PATH.format(diversity), mode='a') as f:
            f.write(f'epoch: {epoch}')
            f.write('\n')
            f.write(generated)
            f.write('\n')
            f.write(DELIMITER)
            f.write('\n')

        print()


# load dataset
with io.open(FILE_PATH, encoding='utf-8') as f:
    text = f.read().lower()
text = text.replace(DELIMITER, '\n\n')
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_chars = []
for i in range(0, len(text) - SEQ_MAX_LEN, STEP):
    sentences.append(text[i:i + SEQ_MAX_LEN])
    next_chars.append(text[i + SEQ_MAX_LEN])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), SEQ_MAX_LEN, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(SEQ_MAX_LEN, len(chars))))
model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH)

model.fit(
    x,
    y,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    callbacks=[print_callback, checkpoint_callback])

model.save(MODEL_FILE_PATH)

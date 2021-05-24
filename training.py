import json
import nltk
import numpy as np
import pickle
import random
import tensorflow as tf

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
word_net_lemmatizer = WordNetLemmatizer()  # The lemmatizer is used so the word 'work' can be translated to have the same meaning as 'works', 'worked' or 'working'.

words = []
tags = []
tags_and_tokenized_words_tuple = []
ignored_words = ['.', ',', '?', '!']

intents = json.loads(open('intents.json').read())

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)  # The tokenization is used to split the words.
        words.extend(tokenized_words)
        tags_and_tokenized_words_tuple.append((intent['tag'], tokenized_words))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

words = [word_net_lemmatizer.lemmatize(word) for word in words if word not in ignored_words]
words = sorted(set(words))

tags = sorted(set(tags))

pickle.dump(tags, open('tags.pkl', 'wb'))
pickle.dump(words, open('words.pkl', 'wb'))

training_set = []
bagged_tags = [0] * len(tags)

for tag_and_tokenized_words in tags_and_tokenized_words_tuple:
    bagged_words = []
    tokenized_words = tag_and_tokenized_words[1]
    lemmatized_words = [word_net_lemmatizer.lemmatize(tokenized_word.lower()) for tokenized_word in tokenized_words]
    for word in words:
        bagged_words.append(1) if word in lemmatized_words else bagged_words.append(0)

    output_row = list(bagged_tags)
    output_row[tags.index(tag_and_tokenized_words[0])] = 1
    training_set.append([bagged_words, output_row])

random.shuffle(training_set)
training_set = np.array(training_set, dtype=object)

training_set_values = list(training_set[:, 0])
training_set_labels = list(training_set[:, 1])

# Artificial Neural Network.
model = Sequential()
model.add(Dense(128, input_shape=(len(training_set_values[0]),), activation=tf.keras.activations.relu))
model.add(Dropout(0.5))
model.add(Dense(64, activation=tf.keras.activations.relu))
model.add(Dropout(0.5))
model.add(Dense(len(training_set_labels[0]), activation=tf.keras.activations.softmax))

# Define the Gradient Descend, the Loss Function and the metrics.
model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-6),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model.
trained_model = model.fit(np.array(training_set_values),
                          np.array(training_set_labels),
                          batch_size=5,
                          epochs=200,
                          verbose=1)

model.save('model.h5', trained_model)

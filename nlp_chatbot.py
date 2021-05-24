import nltk
import json
import numpy as np
import pickle
import random

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

word_net_lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
tags = pickle.load(open('tags.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
model = load_model('model.h5')


def sentence_clean_up(inputSentence):
    tokenized_words = nltk.word_tokenize(inputSentence)
    lemmatized_words = [word_net_lemmatizer.lemmatize(tokenized_word) for tokenized_word in tokenized_words]
    return lemmatized_words


def words_bag_up(inputSentence):
    bagged_words = [0] * len(words)
    sentence_words = sentence_clean_up(inputSentence)
    for sentence_word in sentence_words:
        for word_index, word in enumerate(words):
            if sentence_word == word:
                bagged_words[word_index] = 1

    return np.array(bagged_words)


def tag_prediction(inputSentence):
    error_threshold = 0.25
    returned_list = []

    bagged_words = words_bag_up(inputSentence)
    predictedTags = model.predict(np.array([bagged_words]))[0]

    highest_probability_predicted_tags = [[predicted_tag_index, predicted_tag]
                                          for predicted_tag_index, predicted_tag in enumerate(predictedTags)
                                          if predicted_tag > error_threshold]
    highest_probability_predicted_tags.sort(
        key=lambda highest_probability_predicted_tag: highest_probability_predicted_tag[1],
        reverse=True)

    for predicted_tag in highest_probability_predicted_tags:
        returned_list.append({'tag': tags[predicted_tag[0]], 'tag_probability': str(predicted_tag[1])})
    return returned_list


def get_response(predictedTags):
    predicted_tag = predictedTags[0]['tag']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == predicted_tag:
            returned_response = random.choice(intent['responses'])
            break
    return returned_response


print('The chatbot is up and running. Ask a question.')

while True:
    pattern = input('')
    predicted_tags = tag_prediction(pattern)
    response = get_response(predicted_tags)
    print(response)

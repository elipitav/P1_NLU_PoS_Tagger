import tensorflow as tf
from conllu import parse
import numpy as np

def preprocess_sentences(sentences):
    X_data = []
    y_data = []
    for sentence in sentences:
        parsed_sentence = []
        sentence_labels = []
        for token in sentence:
            if type(token["id"]) == int:
                parsed_sentence.append(token["form"])
                sentence_labels.append(token["upostag"])
        X_data.append(" ".join(parsed_sentence))
        y_data.append(sentence_labels)

    return X_data, y_data

def locate_last_punctuation_mark(sentences_class_labels, limit=None):
    if limit is not None:
        sentences_class_labels = sentences_class_labels[:limit]
    for i, label in enumerate(reversed(sentences_class_labels)):
        if label == "PUNCT":
            return len(sentences_class_labels) - i - 1
    return -1

def truncate_sentence(sentence, sentence_class_labels, max_sentence_length=100):
    last_punctuation_mark = locate_last_punctuation_mark(sentence_class_labels, limit=max_sentence_length)
    sentence = sentence.split()[:max_sentence_length]
    sentence_class_labels = sentence_class_labels[:max_sentence_length]
    if last_punctuation_mark != -1:
        sentence = sentence[: last_punctuation_mark + 1]
        sentence_class_labels = sentence_class_labels[: last_punctuation_mark + 1]
    return " ".join(sentence), sentence_class_labels

class MyTagger(object):
    def __init__(self, train_filename, val_filename, test_filename, max_sentence_num_words=100):
        with open(train_filename, "r", encoding="utf-8") as file:
            train_sentences = parse(file.read())
        with open(test_filename, "r", encoding="utf-8") as file:
            test_sentences = parse(file.read())
        with open(val_filename, "r", encoding="utf-8") as file:
            val_sentences = parse(file.read())
        self.max_sentece_num_words = max_sentence_num_words
            
        self.X_train, self.y_train = preprocess_sentences(train_sentences)
        self.X_test, self.y_test = preprocess_sentences(test_sentences)
        self.X_val, self.y_val = preprocess_sentences(val_sentences)
        
        self.sentence_num_words = [len(sentence.split()) for sentence in self.X_train]
        
        self.sentences_over_max_length = [i for i, num_words in enumerate(self.sentence_num_words) if num_words >self.max_sentence_num_words]
        
        unique_tags = sorted(set(tag for sublist in self.y_train for tag in sublist))
        tag_to_index = {tag: idx for idx, tag in enumerate(unique_tags)}
        self.num_tags = len(tag_to_index)


        for i in self.sentences_over_max_length:
            self.X_train[i], self.y_train[i] = truncate_sentence(self.X_train[i], self.y_train[i], max_sentence_length=max_sentence_num_words)
            
        self.model = None

    def build_model(self, vocabulary_size = 10000, units = 64, output_dim = 50):
        text_vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int", max_tokens=vocabulary_size, output_sequence_length=self.max_sentence_num_words
        )
        text_vectorizer.adapt(self.X_train)
        
        input_layer = tf.keras.layers.Input(shape=(self.max_sentence_num_words,), dtype=tf.int32)
        x = tf.keras.layers.Embedding(input_dim=len(text_vectorizer.get_vocabulary()), output_dim=output_dim)(input_layer)
        x = tf.keras.layers.LSTM(units=units, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_tags, activation="softmax"))(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.summary()
    
    def train(self, optimizer, loss, metrics, batch_size, epochs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(
            np.array(self.X_train), self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(np.array(self.X_val), self.y_val), verbose=True
        )
    
    def evaluate(self):
        self.model.evaluate(np.array(self.X_test), self.y_test)
    
    def predict(self, sentence):
        self.model.predict(np.array(sentence))
    
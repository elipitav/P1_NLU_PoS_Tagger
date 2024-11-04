import tensorflow as tf
from conllu import parse
import numpy as np
import os
from utils import adjust_sentences_length, process_tags, preprocess_sentences, plot_training_history
import json

class MyTagger(object):
    def __init__(self, train_filename, val_filename, test_filename):
        with open(train_filename, "r", encoding="utf-8") as file:
            train_sentences = parse(file.read())
        with open(test_filename, "r", encoding="utf-8") as file:
            test_sentences = parse(file.read())
        with open(val_filename, "r", encoding="utf-8") as file:
            val_sentences = parse(file.read())
            
        self.X_train, self.y_train = preprocess_sentences(train_sentences)
        self.X_test, self.y_test = preprocess_sentences(test_sentences)
        self.X_val, self.y_val = preprocess_sentences(val_sentences)
        
        self.model = None
    
    def preprocess_data(self, max_sentence_num_words=100, with_punctuation=True):
        self.max_sentence_num_words = max_sentence_num_words
        
        # First, we need to truncate the sentences that are longer than the maximum number of words
        adjust_sentences_length(self.X_train, self.y_train, max_sentence_length=self.max_sentence_num_words, with_punctuation=with_punctuation)
        adjust_sentences_length(self.X_val, self.y_val, max_sentence_length=self.max_sentence_num_words, with_punctuation=with_punctuation)
        adjust_sentences_length(self.X_test, self.y_test, max_sentence_length=self.max_sentence_num_words, with_punctuation=with_punctuation)
            
        # Then, we need to convert the labels into numbers and add padding
        self.unique_tags, self.tag_to_index, self.num_tags = process_tags(self.y_train)
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}
        
        # Convert labels into numbers
        y_train_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_train]
        # Add padding to the labels
        self.y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train_indexed, padding="post", maxlen=self.max_sentence_num_words)

        # Same with validation data
        y_val_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_val]
        self.y_val = tf.keras.preprocessing.sequence.pad_sequences(y_val_indexed, padding="post", maxlen=self.max_sentence_num_words)
        
        # Same with test data
        y_test_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_test]
        self.y_test = tf.keras.preprocessing.sequence.pad_sequences(y_test_indexed, padding="post", maxlen=self.max_sentence_num_words)

    def build_model(self, vocabulary_size = 10000, units = 64, output_dim = 50, bidirectional = False):
        self.text_vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int", max_tokens=vocabulary_size, output_sequence_length=self.max_sentence_num_words, standardize = "lower"
        )
        self.text_vectorizer.adapt(self.X_train)
        
        input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
        x = self.text_vectorizer(input_layer)
        x = tf.keras.layers.Embedding(input_dim=len(self.text_vectorizer.get_vocabulary()), output_dim=output_dim, mask_zero = True)(x)
        if bidirectional:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, return_sequences=True))(x)
        else:
            x = tf.keras.layers.LSTM(units=units, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_tags, activation="softmax"))(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.summary()
        
        # Save the initial weights of the model
        self.initial_weights = self.model.get_weights()
    
    def load_model(self, model_filename, model_folder = "./models"):
        # Load the complete model
        model_file = os.path.join(model_folder, f"{model_filename}.keras")
        self.model = tf.keras.models.load_model(model_file)
        print(f"Model loaded from {model_file}")
    
    def save_model(self, model_folder = "./models", model_filename = "trained_model"):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        # Create the folder if it does not exist
        os.makedirs(model_folder, exist_ok=True)

        # Save the complete model
        model_file = os.path.join(model_folder, f"{model_filename}.keras")
        self.model.save(model_file)
        print(f"Model saved in {model_file}")
        
        # Save the training history if provided
        if self.history is not None:
            history_file = os.path.join(model_folder, f"{model_filename}_training_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.history.history, f)
            print(f"Training history saved in {history_file}")
    
    def reset_weights(self):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        # Reset the weights of the model
        if self.initial_weights:
            self.model.set_weights(self.initial_weights)
        else:
            print("Initial weights are not available")
    
    def train(self, optimizer, metrics, batch_size, loss="sparse_categorical_crossentropy", reset_weights = True,  patience = 5, max_epochs = 30):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        # Reset the weights of the model
        if reset_weights:
            self.reset_weights()
        
        # Compile and train the model
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,     
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            np.array(self.X_train), self.y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(np.array(self.X_val), self.y_val), verbose=True, callbacks = [early_stopping]
        )
        
        return self.history
    
    def set_weights(self, weights_filename, weights_folder = "./weights"):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        # Load the weights from the file
        weights_file = os.path.join(weights_folder,f"{ weights_filename}.h5")
        self.model.load_weights(weights_file)
    
    def save_weights(self, weights_folder = "./weights", weights_filename = "saved_weights"):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        # Create the folder if it does not exist
        os.makedirs(weights_folder, exist_ok=True)

        # Save the weights of the model
        weights_file = os.path.join(weights_folder, f"{weights_filename}.h5")
        self.model.save_weights(weights_file)
        print(f"Weights stored in {weights_file}")
        
    def plot_training_history(self):
        plot_training_history(self.history)
    
    def evaluate(self):
        if self.model is None:
            print("The model has not been built yet")
            return

        try:
            self.model.evaluate(np.array(self.X_test), self.y_test)
        except Exception as e:
            print(f"The model has not been compiled yet. Error: {e}")
    
    def predict(self, sentence):
        if self.model is None:
            print("The model has not been built yet")
            return
        
        num_words = len(sentence.split())
        predictions = self.model.predict([sentence])
        predicted_labels = []
        for i in range(len(predictions[0])):
            predicted_index = np.argmax(predictions[0][i])
            predicted_labels.append(self.index_to_tag[predicted_index])

        print(f"Prediction: {predicted_labels[:num_words]}")
    
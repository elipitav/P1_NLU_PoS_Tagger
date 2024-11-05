import tensorflow as tf
from conllu import parse
import numpy as np
import os
from utils import adjust_sentences_length, process_tags, preprocess_sentences, plot_training_history
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
        adjust_sentences_length(
            self.X_train,
            self.y_train,
            max_sentence_length=self.max_sentence_num_words,
            with_punctuation=with_punctuation,
        )
        adjust_sentences_length(
            self.X_val, self.y_val, max_sentence_length=self.max_sentence_num_words, with_punctuation=with_punctuation
        )
        adjust_sentences_length(
            self.X_test, self.y_test, max_sentence_length=self.max_sentence_num_words, with_punctuation=with_punctuation
        )

        # Then, we need to convert the labels into numbers and add padding
        self.unique_tags, self.tag_to_index, self.num_tags = process_tags(self.y_train)
        self.index_to_tag = {idx: tag for tag, idx in self.tag_to_index.items()}

        # Convert labels into numbers
        y_train_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_train]
        # Add padding to the labels
        self.y_train = tf.keras.preprocessing.sequence.pad_sequences(
            y_train_indexed, padding="post", maxlen=self.max_sentence_num_words
        )

        # Same with validation data
        y_val_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_val]
        self.y_val = tf.keras.preprocessing.sequence.pad_sequences(
            y_val_indexed, padding="post", maxlen=self.max_sentence_num_words
        )

        # Same with test data
        y_test_indexed = [[self.tag_to_index[tag] for tag in sublist] for sublist in self.y_test]
        self.y_test = tf.keras.preprocessing.sequence.pad_sequences(
            y_test_indexed, padding="post", maxlen=self.max_sentence_num_words
        )

    def build_model(self, vocabulary_size=10000, units=64, output_dim=50, bidirectional=False):
        self.text_vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int",
            max_tokens=vocabulary_size,
            output_sequence_length=self.max_sentence_num_words,
            standardize="lower",
        )
        self.text_vectorizer.adapt(self.X_train)

        input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
        x = self.text_vectorizer(input_layer)
        x = tf.keras.layers.Embedding(
            input_dim=len(self.text_vectorizer.get_vocabulary()), output_dim=output_dim, mask_zero=True
        )(x)
        if bidirectional:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, return_sequences=True))(x)
        else:
            x = tf.keras.layers.LSTM(units=units, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_tags, activation="softmax"))(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.summary()

        # Save the initial weights of the model
        self.initial_weights = self.model.get_weights()

    def load_model(self, model_filename, model_folder="./models"):
        # Load the complete model
        model_file = os.path.join(model_folder, f"{model_filename}.keras")
        self.model = tf.keras.models.load_model(model_file)
        # Load the training history if available
        history_file = os.path.join(model_folder, f"{model_filename}_training_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                self.history = json.load(f)
        print(f"Model loaded from {model_file}")

    def show_training_log(self):
        epochs = len(self.history["loss"])

        for epoch in range(epochs):
            loss = self.history["loss"][epoch]
            accuracy = self.history["accuracy"][epoch]
            val_loss = self.history["val_loss"][epoch]
            val_accuracy = self.history["val_accuracy"][epoch]

            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

    def save_model(self, model_folder="./models", model_filename="trained_model"):
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
            with open(history_file, "w") as f:
                json.dump(self.history, f)
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

    def train(
        self,
        optimizer,
        metrics,
        batch_size,
        loss="sparse_categorical_crossentropy",
        reset_weights=True,
        patience=5,
        max_epochs=30,
    ):
        if self.model is None:
            print("The model has not been built yet")
            return

        # Reset the weights of the model
        if reset_weights:
            self.reset_weights()

        # Compile and train the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        history = self.model.fit(
            np.array(self.X_train),
            self.y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(np.array(self.X_val), self.y_val),
            verbose=True,
            callbacks=[early_stopping],
        )

        self.history = {
            "loss": history.history["loss"],
            "accuracy": history.history["accuracy"],
            "val_loss": history.history["val_loss"],
            "val_accuracy": history.history["val_accuracy"],
        }

        return self.history

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

    def predict(self, sentences):
        if self.model is None:
            print("The model has not been built yet")
            return

        # Check if the input is a single sentence or a list of sentences
        if isinstance(sentences, str):
            sentences = [sentences]

        predictions = []
        for sentence in sentences:
            splitted_sentence = sentence.split()
            num_words = len(splitted_sentence)

            pred = self.model.predict([sentence])
            predicted_labels = [self.index_to_tag[np.argmax(word)] for word in pred[0]][:num_words]

            predictions.append((splitted_sentence, predicted_labels))

            if len(sentences) == 1:
                print(f"Prediction for '{sentence}':")
                for word, label in zip(splitted_sentence, predicted_labels):
                    print(f"\t{word} -> {label}")

        return predictions

    def plot_confusion_matrix(self):
        if self.model is None:
            print("The model has not been built yet")
            return

        # Initialize the confusion matrix ignoring the padding (class 0)
        confusion = np.zeros((self.num_tags - 1, self.num_tags - 1))

        # Get the predictions for the test set ignoring the padding
        predictions = self.model.predict(np.array(self.X_test))
        predictions = np.argmax(predictions, axis=-1)

        # Compute the confusion matrix
        for i in range(len(self.y_test)):
            for j in range(len(self.y_test[i])):
                if self.y_test[i][j] == 0:
                    continue
                confusion[self.y_test[i][j] - 1, predictions[i][j] - 1] += 1

        # Plot the confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            confusion, annot=True, fmt="g", xticklabels=self.unique_tags, yticklabels=self.unique_tags, cmap="Blues"
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion matrix")

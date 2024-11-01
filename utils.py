import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go

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

def truncate_sentence(sentence, sentence_class_labels, max_sentence_length=100, with_punctuation=True):
    sentence = sentence.split()[:max_sentence_length]
    sentence_class_labels = sentence_class_labels[:max_sentence_length]
    if with_punctuation:
        last_punctuation_mark = locate_last_punctuation_mark(sentence_class_labels, limit=max_sentence_length)
        if last_punctuation_mark != -1:
            sentence = sentence[: last_punctuation_mark + 1]
            sentence_class_labels = sentence_class_labels[: last_punctuation_mark + 1]
    return " ".join(sentence), sentence_class_labels

def adjust_sentences_length(X_data, y_data, max_sentence_length=100, with_punctuation=True):
    for i in range(len(X_data)):
        if len(X_data[i].split()) > max_sentence_length:
            X_data[i], y_data[i] = truncate_sentence(X_data[i], y_data[i], max_sentence_length=max_sentence_length, with_punctuation=with_punctuation)
    return

def process_tags(tags):
    unique_tags = sorted(set(tag for sublist in tags for tag in sublist))
    tag_to_index = {tag: idx + 1 for idx, tag in enumerate(unique_tags)}
    tag_to_index["PAD"] = 0
    num_tags = len(tag_to_index)
    
    return unique_tags, tag_to_index, num_tags

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def calculate_word_frequency(sentences):
    frequencies = dict()

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1
    return frequencies

def add_curve(json_file, name, color, title = "Training", fig = None):
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    val_loss = data.get('val_loss', [])
    loss = data.get('loss', [])
    
    if fig is None:
        fig = go.Figure()
        fig.update_layout(title=title)
    
    fig.add_trace(go.Scatter(
        y=val_loss,
        mode='lines',
        name=f'{name} - val_loss',
        line=dict(dash='dash', color = color)
    ))
    
    fig.add_trace(go.Scatter(
        y=loss,
        mode='lines',
        name=f'{name} - loss',
        line=dict(dash='solid', color = color)
    ))
    
    return fig

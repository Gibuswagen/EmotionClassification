# Importing necessary libraries
import re 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout


from sklearn.metrics import classification_report



# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Read csv file into dataframe
dataframe = pd.read_csv('text.csv')

#Drop the unneeded column, keeping only text and label
dataframe.drop('Unnamed: 0',axis=1,inplace=True)


def expand_abbreviations_and_slangs(text):
    """
    Expands common English abbreviations and internet slang.
    """
    abbreviations = {
        "lol": "laughing out loud",
        "omg": "oh my god",
        "idk": "i do not know",
        "brb": "be right back",
        "gtg": "got to go",
        "ttyl": "talk to you later",
        "bff": "best friends forever",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "rofl": "rolling on the floor laughing",
        "lmao": "laughing my ass off",
        "smh": "shaking my head",
        "np": "no problem",
        "tbh": "to be honest",
        "cant": "cannot",
        "can t": "cannot",
        "wont": "will not",
        "won t": "will not",
        "dont": "do not",
        "don t": "do not",
        "didnt": "did not",
        "didn t": "did not",
        "isnt": "is not",
        "isn t": "is not",
        "arent": "are not",
        "aren t": "are not",
        "you re": "you are",
        "youre": "you are",
        "im": "i am",
        "i m": "i am",
        "let s": "let us",
        "ll": "will",
    }

    # Replace each abbreviation/slang with its full form
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    return text


def preprocess_dataframe(dataframe, text_column):
    """
    Preprocesses a dataframe's text column with the following steps:
    - Converts text to lowercase
    - Removes extra whitespaces
    - Removes HTML tags
    - Removes URLs
    - Removes punctuation
    - Removes stopwords
    - Expands abbreviations and slang
    """
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\bhttp\b|\bhref\b|\bwww\b", "", text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove extra white spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Expand abbreviations
        text = expand_abbreviations_and_slangs(text)

        # Remove stopwords
        text = " ".join(word for word in text.split() if word not in stop_words)
        return text

    # Apply cleaning to the text column
    dataframe[text_column] = dataframe[text_column].apply(clean_text)
    return dataframe


#Apply preprocessing to the dataframe
dataframe = preprocess_dataframe(dataframe, 'text')

# Split dataframe for training and testing
split = int(0.9 * len(dataframe))
train_data = dataframe[:split]  # First 90% for training
test_data = dataframe[split:]  # Last 10% for testing


def prepare_text_sequences(train_texts, test_texts, max_words=50000):
    """
    Prepares text sequences for training and testing by tokenizing and padding.
    
    Args:
        train_texts (list or pd.Series): Training text data.
        test_texts (list or pd.Series): Testing text data.
        max_words (int): Maximum number of words in the vocabulary.

    Returns:
        tokenizer: The trained tokenizer object.
        X_train_padded (ndarray): Padded training sequences.
        X_test_padded (ndarray): Padded testing sequences.
        maxlen (int): Maximum sequence length.
    """
    # Initialize the tokenizer
    tokenizer = Tokenizer(num_words=max_words)

    # Fit tokenizer on training text only
    tokenizer.fit_on_texts(train_texts)

    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_test_seq = tokenizer.texts_to_sequences(test_texts)

    # Get the maximum length of sequences in the training data
    maxlen = max(len(tokens) for tokens in X_train_seq)

    # Pad sequences to ensure uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

    return tokenizer, X_train_padded, X_test_padded, maxlen

# Prepare text sequences
tokenizer, X_train_padded, X_test_padded, maxlen = prepare_text_sequences(
    train_data['text'],
    test_data['text']
)
y_train = train_data['label'].values
y_test = test_data['label'].values


def build_bidirectional_gru_model(max_words, maxlen, embedding_dim=128, gru_units=64, num_classes=6):
    """
    Builds a Bidirectional GRU model for text classification.

    Args:
        max_words (int): Maximum vocabulary size for the embedding layer.
        maxlen (int): Maximum sequence length (input size).
        embedding_dim (int): Dimension of the embedding layer.
        gru_units (int): Number of units in the GRU layer.
        num_classes (int): Number of output classes for classification.

    Returns:
        model: Compiled Bidirectional GRU model.
    """
    # Build the model
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen),
        # Bidirectional GRU layer
        Bidirectional(GRU(gru_units, return_sequences=False)),
        # Dropout for regularization
        Dropout(0.5),
        # Fully connected layer
        Dense(64, activation='relu'),
        # Dropout for regularization
        Dropout(0.5),
        # Output layer with softmax activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define model parameters
max_words = 50000  # Vocabulary size
embedding_dim = 128  # Dimension of word embeddings
gru_units = 64       # Number of GRU units
num_classes = 6      # Number of emotion classes (0-5)

# Build the model
model = build_bidirectional_gru_model(max_words, maxlen, embedding_dim, gru_units, num_classes)

# Summary of the model
print(model.summary())


# Train the model
history = model.fit(
    X_train_padded,   # Padded training sequences
    y_train,          # Training labels
    validation_data=(X_test_padded, y_test),  # Validation on test data
    epochs=5,        # Number of epochs
    batch_size=32,    # Batch size
    verbose=1         # Print progress
)





################
'''
GRAPH PLOTTING
'''
################

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# Predict on the test data
y_pred_probs = model.predict(X_test_padded, verbose=0)  # Predict probabilities
y_pred_classes = y_pred_probs.argmax(axis=1)  # Convert to class labels

# Define class names corresponding to numeric labels
class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Generate the classification report as a dictionary
report_dict = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)

# Convert the dictionary to a pandas DataFrame for easier manipulation
report_df = pd.DataFrame(report_dict).transpose()

# Remove "accuracy", "macro avg", and "weighted avg" rows to focus on class-specific metrics
class_metrics = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

# Plot bar chart for precision, recall, and F1-score
class_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
plt.title('Classification Metrics by Class')
plt.ylabel('Score')
plt.xlabel('Classes')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set the y-axis range to [0, 1] for better comparison
plt.legend(loc='lower right')
plt.show()

# Save to a new CSV file
output_file = "processed_data.csv"
dataframe.to_csv(output_file, index=False)
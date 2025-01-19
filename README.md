## EmotionClassification
Emotion Classification Using Bidirectional GRU 

# Python Version
This project was implemented using the following Python version:
- Python 3.12.2 64-bit

# Third-Party Libraries and Their Versions
The following third-party libraries were used in the implementation:

# Data Preprocessing
- pandas: 2.2.3
- numpy: 2.0.2
- nltk: 3.9.1
- beautifulsoup4: 4.12.3

# Machine Learning Framework
- tensorflow: 2.18.0

# Visualization
- matplotlib: 3.9.2

# Model Evaluation
- scikit-learn: 1.6.0

# How to Run the Code

1. Prerequisites
Before running the code, ensure you have Python 3.12.2 (or compatible version) installed. Install the required third-party libraries using pip:

pip install pandas numpy matplotlib nltk beautifulsoup4 tensorflow scikit-learn

2. Dataset Preparation
Download the dataset from Emotions Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/emotions. (If not in the folder)
Place the dataset (text.csv) in the same directory as the script.

3. How to Run
   
 Run the "main.py". The script will:
    - Read and preprocess the dataset.
    - Split the dataset into training and testing sets.
    - Tokenize and pad the text sequences.
    - Build and train a Bidirectional GRU model.
    - Evaluate the model and generate visualizations for accuracy and loss.
    - Output a classification report and save preprocessed data to a CSV file. (If not commented)

4. Output Files
- processed_data.csv: Preprocessed dataset saved as a CSV file. (Need to uncomment the lines)
- classification_report.txt: Detailed classification metrics.
- Visualizations:
  - Training and Validation Accuracy
  - Training and Validation Loss

5. Results
The script will print:
- Training and validation accuracy and loss per epoch.
- Test accuracy after evaluation.
- Classification report including precision, recall, and F1-score for each class.

6. Notes
- Ensure internet access is available for downloading NLTK stopwords.


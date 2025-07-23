# Spam-Mail-Detection
This project implements a deep learning-based binary classification model to detect spam emails using LSTM (Long Short-Term Memory). The workflow includes full preprocessing, word cloud visualization, tokenization, sequence modeling, and model evaluation.

📚 Dataset
Source: Emails.csv
Columns used:
text: Email body content
label_num: 0 for Non-Spam, 1 for Spam

🧱 Key Steps
✅ Data Preprocessing
Remove "Subject" prefixes
Balance classes by undersampling non-spam messages
Remove punctuation and stopwords

🧠 NLP Pipeline
Tokenization using Tokenizer
Sequence padding to fixed length (100 tokens)
Train/test split (80/20)

📊 Visualizations
Class distribution before and after balancing
Word clouds for spam vs. non-spam emails

🔁 Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=..., output_dim=32),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Loss: Binary Crossentropy (with from_logits=True)
Optimizer: Adam
Callbacks: EarlyStopping + ReduceLROnPlateau

📈 Training Performance
Model is trained for up to 20 epochs
Includes accuracy plots over epochs

Evaluation on test set with:
Test Loss
Test Accuracy

🔧 Technologies Used
TensorFlow / Keras (LSTM, training loop)
NLTK (stopwords)
Pandas, NumPy (data handling)
Seaborn, Matplotlib (visualization)
WordCloud (email word clouds)

✅ Output Example
Test Loss     : 0.18
Test Accuracy : 94.8%

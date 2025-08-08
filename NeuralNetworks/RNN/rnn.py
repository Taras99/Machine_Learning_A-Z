import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/IMDB Dataset.csv")  # update with your path
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

vocab_size = 10000
max_len = 200

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

x_train = tokenizer.texts_to_sequences(train_texts)
x_test = tokenizer.texts_to_sequences(test_texts)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')


def build_model(model_type="rnn"):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, 128, input_length=max_len))
    
    if model_type == "rnn":
        model.add(layers.SimpleRNN(64, activation='tanh'))
    elif model_type == "lstm":
        model.add(layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5))
    elif model_type == "gru":
        model.add(layers.GRU(64, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


results = {}
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

for m_type in ["rnn", "lstm", "gru"]:
    print(f"\nTraining {m_type.upper()} model...")
    model = build_model(m_type)
    history = model.fit(
        x_train, train_labels,
        epochs=10,
        batch_size=64,
        validation_data=(x_test, test_labels),
        callbacks=[early_stop],
        verbose=1
    )
    loss, acc = model.evaluate(x_test, test_labels, verbose=0)
    results[m_type] = acc


print("\nModel Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name.upper()} Accuracy: {accuracy:.4f}")

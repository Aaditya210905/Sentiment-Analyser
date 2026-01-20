import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve,roc_auc_score, auc
import os
import json
import pickle

#1. Load dataset
print("Loading dataset...")
df=pd.read_csv('data/IMDB Dataset.csv')
print(df.head())

#2. Prepare Features and Labels
X=df['review']
y=df['sentiment']

#3. Encoding the target labels
le=LabelEncoder()
y=le.fit_transform(y)

#4. Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#5. Text Preprocessing
print("Preprocessing text data...")
X_train, X_test, tokenizer = preprocess(X_train=X_train, X_test=X_test,tokenizer=None, num_words=16000, max_length=300,
                                         oov_token='<OOV>', padding='pre', truncating='post')
print("Preprocessing completed.")

#6. Define the model
model=Sequential()
model.add(Embedding(16000,128,input_length=300))
model.add(Bidirectional(LSTM(64,dropout=0.3,kernel_regularizer=l2(1e-5),return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32,kernel_regularizer=l2(1e-5),dropout=0.3)))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.build(input_shape=(None, 300))
model.summary()

#7. Train the model with Early Stopping
print("Training the model...")
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

history=model.fit(
    X_train,y_train,epochs=100,batch_size=32,
    validation_data=(X_test,y_test),
    callbacks=[earlystopping]
)
print("Training completed.")

#8. Plot training & validation accuracy and loss values
print("Plotting training history...")
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History with Dropout and L2 Regularization')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

#9. Evaluate Model
print("Evaluating the model...")
def evaluate_model(test, predicted):
    accuracy = accuracy_score(test, predicted) # Calculate Accuracy
    cm = confusion_matrix(test, predicted) # Confusion Matrix
    cr = classification_report(test, predicted) # Classification Report
    precision = precision_score(test, predicted) # Calculate Precision
    recall = recall_score(test, predicted) # Calculate Recall
    roc_auc = roc_auc_score(test, predicted) # Calculate ROC AUC
    return accuracy, cm, cr, precision, recall, roc_auc

y_pred_prob = model.predict(X_test)  # Get predicted probabilities
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

accuracy, cm, cr, precision, recall, roc_auc = evaluate_model(y_test, y_pred)
print("Accuracy", accuracy)
print("Confusion Matrix\n", cm)
print("Classification Report\n", cr)
print("Precision", precision)
print("Recall", recall)
print("ROC AUC", roc_auc)

#10. Plot ROC Curve
print("Plotting ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

#11. Save the model and tokenizer
print("Saving the models...")
tokenizer_path = "models/tokenizer.json"
with open(tokenizer_path, "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

model.save("models/sentiment_model.keras")
print("Models saved successfully.")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


# 🚦 German Traffic Sign Classification

This project implements a Convolutional Neural Network (CNN) to classify images of German traffic signs . The model can help interpret traffic signs for autonomous driving systems.

---

## 📁 Dataset

We use the **GTSRB - German Traffic Sign Recognition Benchmark**, available on Kaggle:

- **Train set**: Images are stored in class-labeled folders (`0` to `42`).
- **Test set**: Images are in one folder, and labels are provided in a CSV file `Test.csv`.

Each image is resized to `64x64` and loaded using `OpenCV`.

---

## 🛠️ Features

- Load and preprocess images
- Resize all to a unified shape (64x64)
- CNN model building and training
- Early stopping and validation split
- Evaluation on test set
- Save predictions to CSV

---

## 🔧 Installation

```bash
pip install numpy pandas matplotlib opencv-python pillow scikit-learn tensorflow
```

> Or simply run in Kaggle or Google Colab where most libraries are preinstalled.

---

## 🚀 How to Run

### 1. Load and Preprocess Data

```python
# Train images
X_train = [...]
y_train = [...]

# Test images from Test.csv
X_test = [...]
y_test = [...]
```

### 2. Split Validation Set (Shuffle)

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.001, shuffle=True, random_state=42)
```

### 3. Build and Train CNN

```python
model = tf.keras.Sequential([
    Conv2D(...),
    MaxPooling2D(...),
    Flatten(),
    Dense(...),
    Dense(43, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=256,
    validation_data=(X_val, y_val),
    callbacks=[early_stop_cb],
    shuffle=True
)
```

### 4. Evaluate and Predict

```python
test_loss, test_acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
```

### 5. Export Predictions

```python
import pandas as pd

image_names = df["Path"].apply(lambda x: x.split("/")[-1])
submission = pd.DataFrame({
    "Image": image_names,
    "TrueLabel": y_test,
    "PredictedLabel": y_pred_labels
})

submission.to_csv("submission.csv", index=False)
```

---

## 📊 Results

- Final Accuracy: `96.3%`
- Early stopping to avoid overfitting
- Confusion matrix and sample predictions visualized in the notebook

---


## 📈 Project Workflow

1. **Load & Preprocess Data**
2. **Build CNN Model**
3. **Train the Model with Validation**
4. **Evaluate Accuracy and Loss**
5. **Visualize Results**

---

## 🔍 Example Output

- Training vs. Validation Accuracy  
- Training vs. Validation Loss  
- Model performance on test set  
- Sample predictions with true vs. predicted labels

---

## 👩‍💻 Author

**Mariam Badr**  
Faculty of Computers & Artificial Intelligence, Cairo University  
[GitHub](https://github.com/Mariam-Badr-MB) – [LinkedIn](https://www.linkedin.com/in/mariambadr13/)

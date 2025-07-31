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

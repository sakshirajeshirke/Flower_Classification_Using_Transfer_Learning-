# Flower_Classification_Using_Transfer_Learning-
A deep learning-based flower classifier using EfficientNetB0 and Streamlit. Upload a flower image and the app predicts its type (Daisy, Dandelion, Rose, Sunflower, Tulip) with confidence scores and details. Built with transfer learning and an interactive UI.

Here's a clean and complete `README.md` tailored for your project **with no demo mode** and based on your `appy.py` and `Flower_TL.ipynb` files:

---

# 🌸 Flower Classifier using Transfer Learning & Streamlit

A deep learning-based flower classification system using **EfficientNetB0** and **Streamlit**. Upload an image of a flower, and the app predicts its type with high confidence and visual insights.

---

## 🔍 Project Overview

This app classifies flowers into 5 categories:
- Daisy 🌼
- Dandelion 🌱
- Rose 🌹
- Sunflower 🌻
- Tulip 🌷

Built using **transfer learning** with EfficientNetB0 and deployed using a custom-styled **Streamlit** UI.

---

## 📁 Files

- `Flower_TL.ipynb`: Model training notebook (includes preprocessing, training, evaluation)
- `appy.py`: Streamlit web app (user interface for classification)
- `flower_model.h5`: Trained EfficientNetB0 model
- `requirements.txt`: Project dependencies

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/flower-classifier-streamlit.git
   cd flower-classifier-streamlit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run appy.py
   ```

4. **Upload a flower image** and view predictions 🎉

---

## 🧠 Model Info

- **Base Model:** EfficientNetB0 (pre-trained on ImageNet)
- **Technique:** Transfer Learning
- **Final Layers:** GlobalAveragePooling, Dense layers, Dropout, Softmax
- **Input Size:** 224x224
- **Framework:** TensorFlow / Keras

---

## 📊 Accuracy

- Training Accuracy: ~97%
- Validation Accuracy: ~92%
- Test Accuracy: ~90%

---

## 💖 Credits

- Dataset: [Kaggle - Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)
- Model: EfficientNetB0 (TensorFlow)
- UI: Streamlit + CSS

---

Let me know if you'd like me to generate the actual `README.md` file content you can download directly.

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
## 📷 Screenshot
![home](https://github.com/user-attachments/assets/4f4712e4-bdab-4705-bb35-71b4a7091178)
![identify](https://github.com/user-attachments/assets/74636ded-73a5-46ab-b02f-06328c118735)
![result](https://github.com/user-attachments/assets/bb95ccfe-0b48-47d0-ab5f-68ab2e94fd38)
![identify3](https://github.com/user-attachments/assets/96ad8bbe-1a01-4fe1-b59e-f4368e65728b)
![result3](https://github.com/user-attachments/assets/41bc2108-64b1-434a-b368-5f007fb971df)





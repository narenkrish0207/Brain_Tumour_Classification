# Brain_Tumour_Classification


🧠 Brain Tumor MRI Image Classification

A Streamlit web application that classifies brain MRI images into different tumor types using a deep learning model trained on medical imaging data.
This project aims to assist with early tumor detection through image-based prediction.

💡 Project Overview

This project demonstrates how deep learning can be applied in healthcare diagnostics.
By uploading a brain MRI image, the model predicts which tumor category it belongs to.

🔍 Tumor Categories

Glioma

Meningioma

Pituitary

No Tumor

⚙️ Tech Stack
Category	Tools Used
Programming Language	Python 🐍
Frameworks	TensorFlow, Streamlit
Libraries	NumPy, Pillow, Pickle
Model Type	InceptionV3 (Pretrained CNN)
Interface	Streamlit Web App
🚀 How It Works

Upload a brain MRI image (JPG/PNG).

The image is preprocessed (resized & normalized).

The trained deep learning model predicts the tumor type.

The app displays:

Predicted class

Confidence score

Probability bar chart

🧠 Model Information

Model Name: MobileNetV2_best.pkl

Input Size: 224×224 pixels

Framework: TensorFlow

Accuracy: ~95% (based on test dataset)

💻 How to Run the App

Clone or download this repository.

Open the project in VS Code or any Python IDE.

Install dependencies:

pip install streamlit tensorflow numpy pillow pickle-mixin

Sure! Here’s the updated README.md with virtual environment setup and pip install instructions clearly included:

⸻



# 😄 Real-Time Emotion Detection using OpenCV & CNN

A Python project for detecting human emotions from facial expressions in real-time using a webcam. It uses **OpenCV** for face detection and a **Convolutional Neural Network (CNN)** trained on the **FER-2013** dataset for emotion classification.

---

## 📁 Project Structure

emotion-detector/
│
├── haarcascade/           # Haar cascade XML for face detection
├── model/                 # Trained emotion detection model (model.h5)
├── dataset/               # FER-2013 dataset (CSV format)
├── train_model.py         # Script to train the CNN model
├── main.py                # Main script for real-time emotion detection
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

---

## 🧰 Installation Guide

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector

✅ Step 2: Set Up Virtual Environment (Recommended)

python -m venv venv

Activate the virtual environment:
	•	On Windows:

venv\Scripts\activate


	•	On Linux/macOS:

source venv/bin/activate



✅ Step 3: Install Dependencies

pip install -r requirements.txt



⸻

📂 Download the Dataset

Download the FER-2013 dataset from Kaggle:
	•	Link: https://www.kaggle.com/datasets/msambare/fer2013

After downloading, place the fer2013.csv file inside the dataset/ folder:

emotion-detector/
└── dataset/
    └── fer2013.csv



⸻

🧠 Train the Model (If Not Already Present)

If model/model.h5 does not exist, run the following command to train it:

python train_model.py

This script will:
	•	Load and preprocess the FER-2013 dataset
	•	Train a CNN model
	•	Save the model as model/model.h5

⸻

🎥 Run the Real-Time Emotion Detection

Once the model is ready and webcam is connected, run:

python main.py

A window will appear showing your webcam feed with real-time emotion predictions overlaid on detected faces.

⸻

🧠 Emotions Detected
	•	Happy
	•	Sad
	•	Angry
	•	Surprise
	•	Neutral
	•	Fear
	•	Disgust

⸻

💡 Use Cases
	•	Mental health and mood tracking
	•	Emotion-aware AI assistants
	•	Smart surveillance and safety systems

⸻

📌 Requirements
	•	Python 3.x
	•	OpenCV
	•	TensorFlow / Keras
	•	NumPy
	•	Pandas
	•	Matplotlib

All are listed in requirements.txt

🙌 Credits
	•	Dataset: FER-2013 - Kaggle
	•	Haarcascade: OpenCV

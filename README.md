# Real-Time Face Recognition & Attendance System

This project is an automated system that uses real-time face recognition to identify individuals and tell their names. The system is built using a pipeline of advanced computer vision models for face detection and recognition, ensuring high accuracy.

<img width="800" height="504" alt="image" src="https://github.com/user-attachments/assets/e37b3480-7aad-430a-be16-dd34198de6b5" />


## Features

-   **Real-Time Detection**: Captures an image and detects faces in real-time.
-   **High-Accuracy Recognition**: Utilizes the pre-trained FaceNet model to generate 128-dimensional facial embeddings for accurate identification.
-   **Automated Attendance Logging**: Can be easily extended to log the name and timestamp of recognized individuals into a CSV file.
-   **Robust Detection Model**: Employs MTCNN (Multi-task Cascaded Convolutional Networks) for reliable face detection in various conditions.
-   **Efficient Classification**: Uses a Support Vector Machine (SVM) classifier trained on facial embeddings for efficient and accurate name prediction.

## Technologies Used

-   **Python 3.x**
-   **Computer Vision**: MTCNN, EfficientNet, SVM, Random Forest
-   **Deep Learning & ML**: Keras, TensorFlow, Scikit-learn
-   **Data Handling**: NumPy, Pickle
-   **Core Model**: FaceNet (for facial embedding generation)

## Project Workflow

The system operates based on the following pipeline:

1.  **Dataset Preparation**: The model is trained on a dataset of images, with each person's images organized in a separate folder.
2.  **Face Detection**: For each image in the dataset, the MTCNN model is used to detect the face and extract the facial region.
3.  **Embedding Generation**: The extracted face is passed to the pre-trained FaceNet model, which generates a unique 128-dimensional vector embedding for that face.
4.  **Classifier Training**: An SVM classifier is trained on the embeddings of all faces in the dataset. A Label Encoder is used to convert person names into numerical labels for training.
5.  **Real-Time Recognition**:
    -   The system captures live images as input and can be further extended to capture videos.
    -   It detects faces using MTCNN.
    -   For each detected face, it generates an embedding using FaceNet.
    -   The trained SVM model predicts the name associated with the embedding with a probability score.

## Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the repository:
```bash
git clone [https://github.com//Mannan-15/FaceRecognitionProject.git](https://github.com/Mannan-15/FaceRecognitionProject.git)
cd your-repo-nam
```

### 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Prepare the Image Dataset:
-   Create a directory named `faces/` in the project folder.
-   Inside this directory, create a sub-directory for each person you want to recognize (e.g., `faces/mannan_golchha/`, `faces/elon_musk/`).
-   Place at least 5-10 clear images of each person in their respective folder.

## (Usage)

To run the application, execute the main Python script from your terminal:
```bash
python facerecognition_project.py
```
-   The first time you run the script, it will automatically process the images in the `faces/` directory, train the SVM model, and save the model files (`svm_model.pkl`, `label_encoder.pkl`, and `faces_embeddings.npz`).
-   On subsequent runs, it will load the saved models and immediately start the real-time recognition.

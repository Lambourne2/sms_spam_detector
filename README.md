# SMS Spam Detector

## Overview
This project implements a machine learning model to classify SMS messages as either spam or ham (legitimate). The model is built using scikit-learn's LinearSVC with TF-IDF vectorization and features a user-friendly Gradio interface for real-time predictions.

## Features
- Text preprocessing and feature extraction using TF-IDF
- Machine learning model for spam detection
- Interactive web interface with Gradio
- Real-time prediction of incoming messages
- Model evaluation metrics and performance analysis

## Dataset
The dataset consists of SMS messages labeled as either "spam" or "ham" (not spam). Each message is preprocessed to extract relevant features for classification.

## Technologies Used
- Python 3.x
- scikit-learn
- Pandas
- NumPy
- Gradio
- Jupyter Notebook

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd sms_spam_detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   If requirements.txt doesn't exist, install the following packages:
   ```bash
   pip install pandas numpy scikit-learn gradio jupyter
   ```

## Usage
1. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook gradio_sms_text_classification.ipynb
   ```

2. For the interactive Gradio interface:
   - Run all cells in the notebook
   - The last cell will launch a local web interface
   - Enter any SMS message in the text box
   - Click "Submit" to see the classification result

## Model Details
The spam detection model uses the following pipeline:
1. **Text Vectorization**: Converts raw text into numerical features using TF-IDF
2. **Classification**: Uses Linear Support Vector Classification (LinearSVC)
3. **Prediction**: Outputs whether the message is "spam" or "ham"

## Model Performance
The model's performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion matrix

## Interactive Demo
An interactive demo is available through Gradio, allowing users to:
- Input any SMS message
- Get instant classification results
- See the confidence score for each prediction

## Files
- `gradio_sms_text_classification.ipynb`: Jupyter notebook with the interactive Gradio interface
- `sms_text_classification_solution.ipynb`: Main notebook with model training and evaluation
- `Resources/`: Directory containing the dataset

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [source]
- Built as part of a machine learning challenge
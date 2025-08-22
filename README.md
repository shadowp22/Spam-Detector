# Spam-Detector

A lightweight Python-based SMS spam detector using NLP, SVM, and Flet for spam/ham classification.

## Overview
Spam-Detector classifies SMS messages as spam or ham using **Natural Language Processing (NLP)** for text preprocessing, a **Support Vector Machine (SVM)** for machine learning-based classification, and a **Flet** GUI for user interaction. It offers a simple interface with theme switching and a standalone executable for offline use. Key features:
- **NLP**: Tokenization, stemming, and stopword removal for message preprocessing.
- **SVM**: Accurate spam/ham classification using a trained model.
- **Flet GUI**: User-friendly interface for instant predictions with light/dark theme support.
- Offline support with bundled models and NLTK data.
- Error handling displayed in the GUI.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AliAminiCode/Spam-Detector.git
   cd Spam-Detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

## Usage
Run the GUI:
```bash
python src/spam_detector_app.py
```
Enter a message (e.g., "Win a free iPhone!" for spam) to see predictions.

To train the model:
```bash
python src/train_spam_classifier.py
```

## Screenshots
Check out Spam-Detector in action:

- **Ham Prediction(Dark Mode)**:  
  <img src="screenshots/ham_prediction.png" alt="Ham Prediction(Dark Mode)">

- **Spam Prediction(Light Mode)**:  
  <img src="screenshots/spam_prediction.png" alt="Spam Prediction(Light Mode)">

## Contribute
Found a bug? Report it at [https://github.com/AliAminiCode/Spam-Detector/issues](https://github.com/AliAminiCode/Spam-Detector/issues).  
Developed by [Ali Amini](mailto:aliamini9728@gmail.com).  
Licensed under the [MIT License](https://github.com/AliAminiCode/Spam-Detector/blob/main/LICENSE).
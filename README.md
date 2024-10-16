# Image Captioning Web Application

Overview
This project is a web application that utilizes deep learning techniques to generate captions for images. Users can upload images from their local device or submit an image URL to receive a descriptive caption.

Features
- Image Upload: Users can upload images directly from their devices.
- Image URL Submission: Users can enter an image URL to generate captions.
- Caption Generation: The application generates captions for the uploaded images using a trained model.

Installation
1. Clone the repository:
   git clone https://github.com/YoussefMArco/Image-Caption-Generator.git
   cd Image-Caption-Generator
2. Install the required dependencies:
   pip install -r requirements.txt

Usage
1. Start the web server:
   python app.py
2. Open a web browser and navigate to http://localhost:5000.
3. Upload an image or enter an image URL and click "Upload and Generate Caption" to view the generated caption.

Example
Here's an example of an image caption generated by the application:

![Generated Caption Example](https://pics.craiyon.com/2023-08-31/2227fdd5564d4c48913b3f7278d6722.webp)

Generated Caption: A dog and a girl are running in the sky.

Evaluation
The model's performance can be evaluated using metrics like BLEU scores. Make sure to refer to the training scripts for more details on performance evaluation.

Acknowledgments
- The project leverages state-of-the-art deep learning models for image captioning.

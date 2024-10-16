# Image Captioning Web Application

Overview
This project is a web application deployed using FastAPI that utilizes deep learning techniques to generate captions for images. Users can upload images from their local device or submit an image URL to receive a descriptive caption.

Features
- Image Upload: Users can upload images directly from their devices.
- Image URL Submission: Users can enter an image URL to generate captions.
- Caption Generation: The application generates captions for the uploaded images using a trained model.

## Model Architecture

The image captioning model consists of two main components:

1. **Image Feature Extractor:** 
   - A pre-trained InceptionV3 model, loaded with ImageNet weights, is used to extract image features. The model's output is the second-to-last layer, capturing a 2048-dimensional vector representation of the image.

2. **Text Sequence Model:** 
   - This is an LSTM-based model that processes the text input, consisting of tokenized words represented via an embedding layer, followed by LSTM layers. The combination of the image features and the text embedding enables the generation of descriptive captions.
   - The image features and text sequences are combined using the `add` layer, and the final output is a vocabulary-sized probability distribution representing the most likely next word in the caption.

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

![Generated Caption Example](https://www.worldatlas.com/upload/4d/95/ce/cda8298a-8220-4e83-8c15-969110c6713d.jpeg)

Generated Caption: the sun and earth

Another Example:

![Generated Caption Example](https://media.istockphoto.com/id/925171128/photo/dog-in-space-suit-hunts-dog-food-hunt.jpg?s=612x612&w=0&k=20&c=clIIhqizJ9LL1mU4vcbqQLlBg-AKVJEtq-LoSTNVieA=)

Generated Caption: a pug dog in an astronaut suit with a space background

Evaluation
The model's performance can be evaluated using metrics like BLEU scores. Make sure to refer to the training scripts for more details on performance evaluation.


Acknowledgments
- The project leverages state-of-the-art deep learning models for image captioning.

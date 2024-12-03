from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from pymongo import MongoClient
import base64
from datetime import datetime
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['smart_closet']
predictions_collection = db['predictions']

# Model Loading
def load_model(repo_id, filename, num_classes):
    """
    Loading the pre-trained model from Hugging Face Hub and modifying the final layer to match the number of classes.
    """
    try:
        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # Initialize the model architecture
        resnet_model = models.resnet50(weights=None)
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
        
        # Load the state dict
        resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        resnet_model.eval()
        return resnet_model
    except Exception as e:
        print(f"Error loading model {filename}: {str(e)}")
        return None

# Load Tensorflow model from Hugging Face Hub
def load_tf_model(repo_id, filename):
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading TF model {filename}: {str(e)}")
        return None

# Define attribute names and class mappings
attribute_names = ["hat", "neckwear", "outer_clothing_cardigan", "upper_clothing_covering_navel"]
class_mappings = {
    "hat": {0: "no hat", 1: "yes hat", 2: "NA"},
    "neckwear": {0: "no neckwear", 1: "yes neckwear", 2: "NA"},
    "outer_clothing_cardigan": {0: "yes cardigan", 1: "no cardigan", 2: "NA"},
    "upper_clothing_covering_navel": {0: "no", 1: "yes", 2: "NA"},
}

# Loading the models from Hugging Face Hub
REPO_ID = "tnalla/smart-closet"

# Load PyTorch models
model = load_model(REPO_ID, "sleeve_model_10.pth", 4)
collar_model = load_model(REPO_ID, "collar_model_10.pth", 7)
lower_length_model = load_model(REPO_ID, "lower_clothing_model_10.pth", 5)

# Load TensorFlow model
tf_model = load_tf_model(REPO_ID, "multi_attribute_classifier.h5")

# Define class labels
class_labels = {0: "sleeveless", 1: "short-sleeve", 2: "medium-sleeve", 3: "long-sleeve"}
collar_class_labels = {0: "V-shape", 1: "square", 2: "round", 3: "standing", 4: "lapel", 5: "suspenders", 6: "NA"}
lower_clothing_length_labels = {0: "three-point", 1: "medium short", 2: "three-quarter", 3: "long", 4: "NA"}

# Image Preprocessing
def preprocess_image(image_bytes):
    """
    Preprocess the uploaded image to match the model's input format.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)

# Image Preprocessing for TensorFlow model
def preprocess_image_tf(image_bytes):
    """
    Preprocess the uploaded image for TensorFlow model.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction
def predict_class(image_tensor, model, class_labels):
    """
    Predicting the class of the given image tensor using the loaded model.
    """
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return class_labels[predicted_class.item()]

# Prediction function for TensorFlow model
def predict_attributes(image_array):
    """
    Predict attributes using TensorFlow model.
    Convert NA predictions to corresponding no values.
    """
    predictions = tf_model.predict(image_array)
    results = {}

    na_to_no_mapping = {
        "hat": "no hat",
        "neckwear": "no neckwear",
        "outer_clothing_cardigan": "no cardigan",
        "upper_clothing_covering_navel": "no"
    }

    for i, attr_name in enumerate(attribute_names):
        class_idx = np.argmax(predictions[i])
        prediction = class_mappings[attr_name][class_idx]
        
        # If prediction is NA, convert to corresponding "no" value
        if prediction == "NA":
            prediction = na_to_no_mapping[attr_name]
            
        results[attr_name] = prediction

    return results

# API Route
@app.route('/predict/sleeve-length', methods=['POST'])
def predict_sleeve_length():
    """
    API endpoint to handle image classification requests.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess the image
        image_tensor = preprocess_image(file.read())

        # Make prediction
        predicted_label = predict_class(image_tensor, model, class_labels)

        # Return the result
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/collar-type', methods=['POST'])
def predict_collar_type():
    """
    API endpoint to handle image classification requests.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess the image
        image_tensor = preprocess_image(file.read())

        # Make prediction
        predicted_label = predict_class(image_tensor, collar_model, collar_class_labels)

        # Return the result
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/lower-length', methods=['POST'])
def predict_lower_length():
    """
    API endpoint to predict lower clothing length from an uploaded image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess the image
        image_tensor = preprocess_image(file.read())

        # Make prediction
        predicted_label = predict_class(image_tensor, lower_length_model, lower_clothing_length_labels)

        # Return the result
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Combined API Route
@app.route('/predict', methods=['POST'])
def predict_both():
    """
    API endpoint to predict all attributes from an uploaded image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the image file
        image_bytes = file.read()

        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Preprocess the image for PyTorch models
        image_tensor = preprocess_image(io.BytesIO(image_bytes).read())

        # Make predictions with PyTorch models
        sleeve_prediction = predict_class(image_tensor, model, class_labels)
        collar_prediction = predict_class(image_tensor, collar_model, collar_class_labels)
        lower_length_prediction = predict_class(image_tensor, lower_length_model, lower_clothing_length_labels)

        # Preprocess and predict with TensorFlow model
        image_array = preprocess_image_tf(image_bytes)
        tf_predictions = predict_attributes(image_array)

        # Combine all predictions
        all_predictions = {
            'sleeve_length': sleeve_prediction,
            'collar_type': collar_prediction,
            'lower_length': lower_length_prediction,
            **tf_predictions
        }

        # Check for similar items
        similar_items = list(predictions_collection.find({
            'predictions.sleeve_length': sleeve_prediction,
            'predictions.collar_type': collar_prediction,
            'predictions.lower_length': lower_length_prediction,
            'predictions.hat': tf_predictions['hat'],
            'predictions.neckwear': tf_predictions['neckwear'],
            'predictions.outer_clothing_cardigan': tf_predictions['outer_clothing_cardigan'],
            'predictions.upper_clothing_covering_navel': tf_predictions['upper_clothing_covering_navel']
        }, {'_id': 0}))

        if similar_items:
            return jsonify({
                'message': 'Similar items found in your closet',
                'current_predictions': all_predictions,
                'similar_items': similar_items,
                'count': len(similar_items)
            }), 200

        # Create document for MongoDB
        prediction_doc = {
            'filename': file.filename,
            'image': image_base64,
            'predictions': all_predictions,
            'timestamp': datetime.utcnow()
        }

        # Store in MongoDB
        predictions_collection.insert_one(prediction_doc)

        # Return the predictions
        return jsonify({
            'success': True,
            'message': 'No similar items found. New item has been saved.',
            'filename': file.filename,
            'predictions': all_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/save', methods=['POST'])
def save_prediction():
    """
    API endpoint to predict and save the image with its predictions directly to MongoDB.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the image file
        image_bytes = file.read()

        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Preprocess the image for PyTorch models
        image_tensor = preprocess_image(io.BytesIO(image_bytes).read())

        # Make predictions with PyTorch models
        sleeve_prediction = predict_class(image_tensor, model, class_labels)
        collar_prediction = predict_class(image_tensor, collar_model, collar_class_labels)
        lower_length_prediction = predict_class(image_tensor, lower_length_model, lower_clothing_length_labels)

        # Preprocess and predict with TensorFlow model
        image_array = preprocess_image_tf(image_bytes)
        tf_predictions = predict_attributes(image_array)

        # Combine all predictions
        all_predictions = {
            'sleeve_length': sleeve_prediction,
            'collar_type': collar_prediction,
            'lower_length': lower_length_prediction,
            **tf_predictions
        }

        # Create document for MongoDB
        prediction_doc = {
            'filename': file.filename,
            'image': image_base64,
            'predictions': all_predictions,
            'timestamp': datetime.utcnow()
        }

        # Store in MongoDB
        result = predictions_collection.insert_one(prediction_doc)

        # Return success response
        return jsonify({
            'success': True,
            'message': 'Item saved successfully',
            'filename': file.filename,
            'predictions': all_predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get_items', methods=['GET'])
def get_items():
    """
    API endpoint to fetch all items from predictions collection.
    Returns a list of items with their images and predictions.
    """
    try:
        # Fetch all items from predictions collection
        items = list(predictions_collection.find({}, {'_id': 0}))

        return jsonify({
            'success': True,
            'items': items
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/recommend', methods=['POST'])
def recommend_items():
    """
    API endpoint to get top 3 recommendations based on given attributes.
    If no exact matches found, returns items matching at least 3 attributes.
    """
    try:
        # Get attributes from request
        attributes = request.json
        if not attributes:
            return jsonify({
                'success': False,
                'error': 'No attributes provided'
            }), 400

        required_attributes = [
            'collar_type', 'hat', 'lower_length', 'neckwear',
            'outer_clothing_cardigan', 'sleeve_length', 'upper_clothing_covering_navel'
        ]

        # Validate all required attributes are present
        missing_attrs = [attr for attr in required_attributes if attr not in attributes]
        if missing_attrs:
            return jsonify({
                'success': False,
                'error': f'Missing required attributes: {", ".join(missing_attrs)}'
            }), 400

        # First try exact match
        exact_match_pipeline = [
            {
                '$match': {
                    '$and': [
                        {'predictions.' + attr: value} 
                        for attr, value in attributes.items()
                    ]
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'filename': 1,
                    'image': 1,
                    'predictions': 1,
                    'timestamp': 1,
                    'match_count': {'$literal': len(attributes)}  # All attributes match
                }
            },
            {'$limit': 3}
        ]

        recommendations = list(predictions_collection.aggregate(exact_match_pipeline))

        # If no exact matches, try partial matches
        if not recommendations:
            # Create conditions for each attribute
            conditions = [
                {'predictions.' + attr: value} 
                for attr, value in attributes.items()
            ]

            partial_match_pipeline = [
                {
                    '$project': {
                        '_id': 0,
                        'filename': 1,
                        'image': 1,
                        'predictions': 1,
                        'timestamp': 1,
                        'match_count': {
                            '$sum': [
                                {'$cond': [{'$eq': ['$predictions.' + attr, value]}, 1, 0]}
                                for attr, value in attributes.items()
                            ]
                        }
                    }
                },
                {
                    '$match': {
                        'match_count': {'$gte': 3}  # At least 3 attributes match
                    }
                },
                {
                    '$sort': {'match_count': -1}  # Sort by number of matching attributes
                },
                {'$limit': 3}
            ]

            recommendations = list(predictions_collection.aggregate(partial_match_pipeline))

        if not recommendations:
            return jsonify({
                'success': True,
                'message': 'No items found with at least 3 matching attributes',
                'recommendations': []
            })

        return jsonify({
            'success': True,
            'count': len(recommendations),
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)

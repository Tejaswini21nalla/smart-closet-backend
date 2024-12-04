# Smart Closet Backend

A Flask-based backend service for the Smart Closet application that provides clothing attribute classification using deep learning models.

## Features

- Sleeve length classification
- Collar type detection
- Lower clothing length classification
- Multi-attribute detection
- Image storage and retrieval
- Clothing recommendations based on attributes

## Prerequisites

- Python 3.9+
- MongoDB Atlas account
- Docker (optional for containerization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tejaswini21nalla/smart-closet-backend.git
cd smart-closet-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MongoDB Atlas:
- Create a MongoDB Atlas account and set up a free cluster
- Add your IP to the network access list
- Create a database user with the necessary permissions
- Obtain the connection string and update your `.env` file:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/<database-name>?retryWrites=true&w=majority
```

## Running the Application Locally

To run the application locally without Docker:

1. Ensure your virtual environment is activated:
```bash
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

2. Set up MongoDB:
- Ensure MongoDB is installed and running locally.
- By default, the application connects to `mongodb://localhost:27017/`.
- If needed, update the `.env` file with your local MongoDB URI:
```env
MONGODB_URI=mongodb://localhost:27017/your_database_name
```

3. Start the Flask server:
```bash
flask run --host=0.0.0.0 --port=5000
```

The server will start on `http://localhost:5000` in development mode.

Alternatively, you can start the Flask server using Python:
```bash
python app.py
```

## Dataset

The models used in this application are trained on the DeepFashion dataset, which is a large-scale clothes database containing diverse fashion images. This dataset is widely used in research for tasks such as clothing attribute prediction, fashion item retrieval, and more.

The DeepFashion dataset is provided by MMLab@NTU, affiliated with S-Lab, Nanyang Technological University, and SenseTime Research.

For more information about the dataset, you can visit the [DeepFashion-MultiModal GitHub repository](https://github.com/yumingj/DeepFashion-MultiModal).

## Docker Setup

1. Build the Docker image:
```bash
docker build -t smart-closet-backend .
```

2. Run the Docker container:
```bash
docker run -d -p 80:80 -e MONGODB_URI="your_mongodb_uri" --name smart-closet-backend smart-closet-backend
```

## API Endpoints

### 1. Predict All Attributes
- **Endpoint**: `/predict/both`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: All clothing attributes including sleeve length, collar type, and lower clothing length

### 2. Save Prediction
- **Endpoint**: `/save-prediction`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: Saved prediction ID and all clothing attributes
- **Description**: Saves the image and its predictions to MongoDB

### 3. Get All Items
- **Endpoint**: `/items`
- **Method**: GET
- **Returns**: List of all saved items with their predictions

### 4. Get Recommendations
- **Endpoint**: `/recommend`
- **Method**: POST
- **Input**: JSON with desired attributes
- **Returns**: Top 3 matching items from the database

### 5. Recommend Items with Similarity Measures
- **Endpoint**: `/recommend/similarity`
- **Method**: POST
- **Input**: JSON object with desired attributes
- **Returns**: Top 3 items with the highest similarity scores based on multiple similarity measures

This API endpoint calculates similarity scores using cosine similarity, Jaccard similarity, and Hamming distance for each item in the database. It ranks items by highest cosine and Jaccard similarity and lowest Hamming distance, returning the top 3 items with the best similarity scores.

Example Request:
```json
{
  "collar_type": "V-shape",
  "hat": "yes hat",
  "lower_length": "long",
  "neckwear": "no neckwear",
  "outer_clothing_cardigan": "no",
  "sleeve_length": "long-sleeve",
  "upper_clothing_covering_navel": "yes"
}
```

Example Response:
```json
{
  "success": true,
  "count": 3,
  "recommendations": [
    {
      "filename": "item1.jpg",
      "image": "base64encodedimage",
      "predictions": {
        "collar_type": "V-shape",
        "hat": "yes hat",
        "lower_length": "long",
        "neckwear": "no neckwear",
        "outer_clothing_cardigan": "no",
        "sleeve_length": "long-sleeve",
        "upper_clothing_covering_navel": "yes"
      },
      "cosine_similarity": 0.95,
      "jaccard_similarity": 0.85,
      "hamming_distance": 1,
      "timestamp": "2023-10-15T12:34:56"
    },
    ...
  ]
}
```

## Model Information

The application uses multiple deep learning models for classification:

### ResNet50-based Models
- Sleeve length model: 4 classes (sleeveless, short-sleeve, medium-sleeve, long-sleeve)
- Collar type model: 7 classes (V-shape, square, round, standing, lapel, suspenders, NA)
- Lower clothing length model: 5 classes (three-point, medium short, three-quarter, long, NA)

### Multi-Attribute Classifier
- TensorFlow-based model for detecting multiple attributes simultaneously
- Attributes detected:
  - Hat presence (no hat, yes hat)
  - Neckwear presence (no neckwear, yes neckwear)
  - Cardigan presence (yes cardigan, no cardigan)
  - Navel coverage (no, yes)

All models are hosted on Hugging Face Hub: [tnalla/smart-closet](https://huggingface.co/tnalla/smart-closet/tree/main)

## Example Usage

### Python Request Example
```python
import requests

# Predict all attributes
url = 'http://localhost:80/predict/both'
files = {'image': open('shirt.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful operation
- 400: Invalid input
- 500: Server error

Error responses include a message explaining the error.

## Development

The application is built with:
- Flask for the REST API
- PyTorch for deep learning models
- MongoDB for data storage
- Hugging Face Hub for model hosting

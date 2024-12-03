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

2. Start the Flask server:
```bash
flask run --host=0.0.0.0 --port=5000
```

The server will start on `http://localhost:5000` in development mode.

Alternatively, you can start the Flask server using Python:
```bash
python app.py
```

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

### 1. Predict Sleeve Length
- **Endpoint**: `/predict/sleeve-length`
- **Method**: POST
- **Input**: Form data with 'image' field containing the image file
- **Returns**: Sleeve length classification (sleeveless, short-sleeve, medium-sleeve, long-sleeve)

### 2. Predict Collar Type
- **Endpoint**: `/predict/collar-type`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: Collar type classification (V-shape, square, round, standing, lapel, suspenders)

### 3. Predict Lower Clothing Length
- **Endpoint**: `/predict/lower-length`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: Lower clothing length classification (three-point, medium short, three-quarter, long)

### 4. Predict All Attributes
- **Endpoint**: `/predict/both`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: All clothing attributes including sleeve length, collar type, and lower clothing length

### 5. Save Prediction
- **Endpoint**: `/save-prediction`
- **Method**: POST
- **Input**: Form data with 'image' field
- **Returns**: Saved prediction ID and all clothing attributes
- **Description**: Saves the image and its predictions to MongoDB

### 6. Get All Items
- **Endpoint**: `/items`
- **Method**: GET
- **Returns**: List of all saved items with their predictions

### 7. Get Recommendations
- **Endpoint**: `/recommend`
- **Method**: POST
- **Input**: JSON with desired attributes
- **Returns**: Top 3 matching items from the database

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

# Predict sleeve length
url = 'http://localhost:80/predict/sleeve-length'
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

## A2 Language Model: Harry Potter Text Generation

Project Overview
This project implements an LSTM (Long Short-Term Memory) neural network trained to generate text in the style of "Harry Potter and the Sorcerer's Stone." The model learns from text sequences to predict the next word, allowing it to complete sentences based on user input.

Key Features:

- Model: A custom 2-layer LSTM with 1024 hidden units and 0.65 dropout for regularization.
- Dataset: The model uses a vocabulary of 2,154 unique words derived from the first Harry Potter book.
- Web App: A user-friendly dashboard built with Dash that displays generated text.
- emperature Sampling: The application generates 10 unique results at once by varying the "temperature" (creativity level) from 0.5 to 1.1.

Follow these steps to run the application on your local machine:

- Install Dependencies pip install torch dash
- Run the Application Navigate to the project folder in your terminal and run: python app/app.py
- Access the Interface Open your web browser and go to: http://127.0.0.1:8050/

How to Use:

- Enter a starting phrase (e.g., "Harry Potter is") into the text box.
- Click Generate.
- The app will display 10 different completions:
- Low Temperature (0.5 - 0.7): Produces logical, safe predictions.
- High Temperature (0.8 - 1.1): Produces diverse and creative predictions.

File Structure:

- A2_Language_Model.ipynb: The notebook used to train and evaluate the model.
- app/app.py: The main script for the web application.
- app/lstm.py: The model architecture definition.
- models/: Contains the saved model weights (.pt) and vocabulary (.pkl).
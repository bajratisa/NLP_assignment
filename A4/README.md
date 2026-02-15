# NLP Assignment 4: Do you AGREE?

## Project Description
This project is my submission for the Natural Language Understanding (NLU) assignment. The goal of this project is to understand the inner workings of Transformer models by building a Bidirectional Encoder Representations from Transformers (BERT) model completely from scratch. 

After pre-training the custom BERT model on a subset of English Wikipedi, I adapted it into a Sentence-BERT (Siamese Network) architecture. This allows the model to perform Natural Language Inference (NLI) by comparing two sentences and determining their semantic relationship using a classification objective function (Softmax Loss).

## Technologies Used
* **Python**
* **PyTorch:** Used for building and training the neural network architectures from scratch.
* **Dash:** Used to build the interactive web application interface.
* **Hugging Face (`transformers` & `datasets`):** Used strictly for data loading (WikiText and SNLI) and basic tokenization
* **Scikit-learn:** Used for evaluating the model and generating the classification report.

## What the Web App Does
The final deliverable includes a simple web application built with Dash. The app demonstrates the capabilities of the trained Sentence-BERT model. 

It provides two input boxes for a user to type in:
1. **Premise:** A starting sentence (e.g., "A man is playing a guitar on stage.").
2. **Hypothesis:** A following sentence (e.g., "The man is performing music.").

The model processes both sentences and predicts their relationship label:
* **Entailment:** The hypothesis is definitely true based on the premise.
* **Neutral:** The hypothesis might be true, but it's not guaranteed.
* **Contradiction:** The hypothesis is definitely false based on the premise.

## Sample Classification Report
As part of the evaluation step, the model generates a performance metric report based on the NLI task. Below is the sample format of the classification report evaluated in this project:

## Classification Report

| Class          | Precision | Recall | F1-Score | Support |
|----------------|----------:|-------:|---------:|--------:|
| Entailment     | 0.61      | 0.75   | 0.68     | 3368    |
| Neutral        | 0.62      | 0.63   | 0.63     | 3219    |
| Contradiction  | 0.69      | 0.51   | 0.59     | 3237    |
| Accuracy       |           |        | 0.63     | 9824    |
| Macro Avg      | 0.64      | 0.63   | 0.63     | 9824    |
| Weighted Avg   | 0.64      | 0.63   | 0.63     | 9824    |


## How to Run the Project

Once the notebook has finished running and the model weights are saved, you can launch the Dash web app. In your terminal, navigate to the root folder of this project and run:

python app/app.py

Then, open your web browser and go to the local address provided in the terminal (usually http://127.0.0.1:8050/) to test the model!
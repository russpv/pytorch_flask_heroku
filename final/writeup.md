# NLP Neural Net Product Classifier
Russell Peavey

_Neural net that classifies products into hundreds of classes based on short ERP text description_

## Abstract

The problem: classifying product materials in catalog master data is typically challenging at B2B industrial manufacturers, a landscape characterized by large product lists and fragmented customer bases where repeat sales make up the bulk of business. This means you have to get it right, and often it isn't. Incorrect classifications are difficult to fix once committed to the ERP. These errors then propagate to downstream analytics. 

One root issue is contextual information is not presented to data owners at the time product classifications are made. Owners can forget the heuristic classification rules (e.g. product A belongs in class M because of property X) or misread sometimes ambiguous or erroneous material text descriptions.

Solution: assist data owners with predicted default classification codes prior to commits as ERP master data. 

Raw text is loaded into pandas dataframes and preprocessed prior to modeling to do basic tokenization on the corpus of descriptions, adding PAD, UNK and END as needed. Dictionaries are built with an optional frequency threshold depending on training data size.

## Design

The classifier is trained offline and uploaded to a Flask server on Heroku. Because the data is consumed by the client organization in Excel, an M-language workbook query fetches model predictions into a spreadsheet.

## Data

In this real-world scenario, little existing training data was available to classify the tens of thousands of materials into hundreds of classes. A random sampling of less than 1,000 materials was sent to a handful of individuals for manual spreadsheet completion given the calendar time allotted and used as training data.

Raw data is taken at a point in the material creation pipeline where a text description is created by the product lab and is not yet submitted for commercial production. The text description is only 40 alphanumeric characters long. 

## Model Selection

Several approaches were considered:

- Random forest decision tree
- Naive Bayes n-gram (multinomial, Bernoulli)
- 1d CNN
- RNN (GRU)
- Transformer with self-attention

Non-NLP classifiers were discarded to save implementation time. Accessing the proprietary product recipe database to obtain product features is time consuming, and would also entail subsequent feature engineering.

Unigram and bigram models were tested as language model baselines.

PyTorch Dataloaders are constructed with dictionaries that feed the neural models in batches. The models comprise an embedding layer, bidirectional GRU layers where the last hidden time step is concatenated and passed to a regularizing dropout layer prior to a linear output layer. Random word embedding vectors are initialized for learning during training. Softmax is applied during training: done with the ADAM optimizer and cross-entropy loss.

## Evaluation Method

6-fold cross validation is done to randomize the splits and the accuracy scores are averaged (making public shortly).

## Tools

- Data Cleaning
  - Train/test splitting with R to CSV
  - Preprocessing and tokenization with Python
- Model Development and Testing
  - Build, evaluation, serialization with PyTorch
- API Interface
  - Flask
- Web App Deployment
  - Heroku
- Client access
  - Power BI M Language POST query

## Communication

A brief presentation of the modeling process and results will be delivered.

The Flask app can be POSTed to here: https://pytorch_flask_heroku.herokuapp.com/predict

## Version 2 Features

- Better testing management with FastAPI instead of Flask
- Compare word embedding methods and try GenSIM FastText
- Add scripted k-fold eval
- Add web form endpoint

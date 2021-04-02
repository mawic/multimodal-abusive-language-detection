# Accompanying Code for submission 663 to ECML/PKDD 21

# Project Installation Instructions
Set up a conda environment using requirements.txt and following the additional installation-instructions.txt 

# Structure

The project root folder mainly divides itself into five parts:

* **Preprocessing**, containing a brief overview of some initial data adjustments necessary for the vocabulary, the network and the explainability of the network

* the individual model components, mainly **TweetClassifier**, **TweetHistory** and **TweetNetwork** containing the relevant models for each component

* the **joint\_model.py** file aggregating the individual components into one joint model

* the **SHAP** folder containing the explainability methods

* a folder containing several **jupyter notebooks**: one for each dataset to initialize model, data and training as well as results computations; one for each dataset to derive the GraphSAGE visualizations and an exemplary workbook to showcase the SHAP computations

* **utils** containing support methods required to run the model

* **dataset parser** for each dataset used to generate the training data, validation data and test data for the 


# Model Flow

An overview of the model flow is provided in the visualization below:
![alt text](image.png "Overview")

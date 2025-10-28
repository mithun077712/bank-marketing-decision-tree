# bank-marketing-decision-tree
Decision tree classifier for the UCI Bank Marketing "additional" dataset predicting customer subscription behavior. Includes data preprocessing, model training, evaluation, and feature importance visualization using Python and scikit-learn.  This concise name follows GitHub best practices with lowercase letters and hyphens for readability. 
Bank Marketing Decision Tree Classifier
This project implements a Decision Tree classifier using the UCI Bank Marketing "additional" dataset. The goal is to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral data.

Project Overview
Dataset: UCI Bank Marketing Dataset

Input: Customer attributes including age, job, marital status, campaign data, and more.

Output: Predict if a customer subscribes (yes or no)

Model: Decision Tree Classifier built and evaluated using Python’s scikit-learn.

Features
Data preprocessing including label encoding of categorical variables.

Train-test split for model validation.

Model training and prediction.

Performance evaluation using accuracy, confusion matrix, and classification report.

Visualization of feature importances and decision tree structure.

Printing dataset documentation from the provided bank-additional-names.txt file.

How to Run
Download the dataset bank-additional-full.csv and the feature description bank-additional-names.txt from the UCI ML repository.

Update the file paths in the Python script (csv_path and names_path) to point to where the files are saved.

Install required Python libraries if not installed:

text
pip install pandas scikit-learn matplotlib
Run the Python script:

text
python skillcraft3.py
Check the console for dataset information, evaluation metrics, and view plots for feature importance and the decision tree.

Requirements
Python 3.7+

pandas

scikit-learn

matplotlib

Results
The model will output the accuracy score, confusion matrix, and detailed classification report on the test data.

The top features influencing the prediction will be displayed in a bar chart.

Decision Tree plot visualizes the model's split nodes for interpretability.

License
Specify your preferred license here, e.g., MIT License.


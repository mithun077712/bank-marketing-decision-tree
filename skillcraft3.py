names_path = r"C:\Users\DELL\OneDrive\Documents\bank-additional\bank-additional\bank-additional-names.txt"
csv_path = r"C:\Users\DELL\OneDrive\Documents\bank-additional\bank-additional\bank-additional-full.csv"

print("=== Dataset Description ===")
with open(names_path, 'r') as file:
    names_content = file.read()
    print(names_content)
print("==========================\n")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv(csv_path, sep=';')

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object' and column != 'y':
        df[column] = le.fit_transform(df[column])

df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No','Yes'], filled=True, max_depth=3)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the data
data = pd.read_csv('emails.csv')
print(data.head())

# 2. Prepare the data
X = data['email_text']  # Features (email content)
y = data['label']       # Labels (phishing or legitimate)

# 3. Turn text into numbers
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X)

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# 5. Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# 6. Test the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn import tree
import matplotlib.pyplot as plt

# 7. Visualize the Decision Tree (prettier version)
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(
    model,
    filled=True,
    rounded=True,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=model.classes_,
    fontsize=10
)
plt.show()
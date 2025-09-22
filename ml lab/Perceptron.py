import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("spam.csv")
x = data.drop("Label (Spam=1 / Not Spam=0)", axis=1)  # Features
y = data["Label (Spam=1 / Not Spam=0)"]               # Target
#stratify ensures that the class distribution (proportion of labels)
#in the train and test sets is the same as in the original dataset.
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2,
                                      random_state=42, stratify=y)
#eta0 is the learning rate (step size) used by the perceptron during training.
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(xtr, ytr)
ypr = model.predict(xte)

cm = confusion_matrix(yte, ypr)
acc = accuracy_score(yte, ypr)

print("Classification Report:\n")
print(classification_report(yte, ypr, target_names=['Not Spam','Spam']))
print(f"Accuracy: {acc*100:.2f}%")

# Plot heatmap
plt.figure(figsize=(4,3))
#annotation actual numbers in cells
#fmt format in d for decimal integer default it is float
#cmap color map as blue default it is yellow purple 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam','Spam'],
            yticklabels=['Not Spam','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


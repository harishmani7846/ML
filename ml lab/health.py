import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv('/home/kcet/health.csv')


df['Gender']=LabelEncoder().fit_transform(df['Gender'])


x = df[['Age', 'Gender', 'Bmi', 'BP', 'Cholesterol']]
y = df['Condition']


scaler = StandardScaler()
xscale = scaler.fit_transform(x)


xtr, xte, ytr, yte = train_test_split(xscale, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(xtr, ytr)


ypr = model.predict(xte)
yprob = model.predict_proba(xte)[:, 1]

print("Accuracy score:", accuracy_score(yte, ypr))
print("Classification Report:\n", classification_report(yte, ypr, zero_division=1))


cm = confusion_matrix(yte, ypr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


new = pd.DataFrame([[60, 1, 27, 130, 200]], columns=['Age', 'Gender', 'Bmi', 'BP', 'Cholesterol'])
newscale = scaler.transform(new)
newcondition = model.predict_proba(newscale)[0][1]
print(f"Probability of developing the condition: {newcondition:.2f}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

da=pd.read_csv("/home/student/stud.csv")

pf=pd.DataFrame(da)

x=pf[['Studying hours','Attendance']]
y=pf['Result']

clf=DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(x,y)
plt.figure(figsize=(8,6))

plot_tree(clf,feature_names=['Studying hours','Attendance'],class_names=['0','1'],filled=True)
plt.show()

new=[[5,85]]
pred=clf.predit(new)
print("Prediction for new student:","1" if pred[0] == 1 else "0")

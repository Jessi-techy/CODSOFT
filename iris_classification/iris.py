import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix

iris_dataset = pd.read_csv("C:/Users/JAYA SREE/Downloads/IRIS.csv")
print(iris_dataset.head(6))
pd.set_option('future.no_silent_downcasting', True)
iris_dataset_new = iris_dataset.replace({'species':{ 'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}})
iris_dataset_new = iris_dataset_new.infer_objects(copy=False)

X=iris_dataset_new.drop(columns=["species"],axis=1)
y=iris_dataset_new["species"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
model = LogisticRegression()
model.fit(X_train,y_train)
training_prediction = model.predict(X_train)
print("Training Accuracy : ",accuracy_score(y_train,training_prediction))
testing_prediction = model.predict(X_test)
print("Testing Accuracy : ",accuracy_score(y_test,testing_prediction))
print("\nClassification Report:\n", classification_report(y_test, testing_prediction))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, testing_prediction))

conf_matrix = confusion_matrix(y_test, testing_prediction)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=['Setosa', 'Versicolor', 'Virginica'], yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

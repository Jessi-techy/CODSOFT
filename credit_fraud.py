import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
credit_dataset = pd.read_csv("C:/Users/JAYA SREE/OneDrive/Documents/creditcard.csv")
# print(credit_dataset.head(20))
# print(credit_dataset.shape)
a=credit_dataset['Class'].value_counts()
# print(a)
legit = credit_dataset[credit_dataset.Class == 0]
fraud = credit_dataset[credit_dataset.Class == 1]
# print(legit.shape)
a1=legit.Amount.describe()
# print(a1)
classify =credit_dataset.groupby('Class').mean()
# print(classify)
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample,fraud],axis=0)
# print(new_dataset['Class'].value_counts())
new_classify = new_dataset.groupby('Class').mean()
# print(new_classify)
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']


X_train , X_test , Y_train ,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train,Y_train)
training_prediction = model.predict(X_train)

print("Accuracy for training: ",accuracy_score(Y_train,training_prediction))
testing_prediction = model.predict(X_test)

print("Accuracy for training: ",accuracy_score(Y_test,testing_prediction))
print("Confusion Matrix:\n", confusion_matrix(Y_test, testing_prediction))
print("Classification Report:\n", classification_report(Y_test, testing_prediction))


cm = confusion_matrix(Y_test, testing_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump


df = pd.read_csv('./Iris.csv')
print(df.head())

iris = df.drop("Id", axis= 1)
x = iris.drop("Species", axis=1)
y = iris["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
dump(knn, 'knn_model.joblib')


y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
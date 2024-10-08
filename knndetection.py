
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

bc = datasets.load_breast_cancer()

x = bc.data
y = bc.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train, y_train)

y_prediction = classifier.predict(x_test)

acc = metrics.accuracy_score(y_test, y_prediction)

print(acc)

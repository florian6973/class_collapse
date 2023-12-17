import class_collapse.data.house_dataset as house_dataset
from sklearn.model_selection import train_test_split

# knn
from sklearn.neighbors import KNeighborsClassifier

X, y_coarse, y_fine = house_dataset.get_house_dataset(None)

X_train, X_test, y_train, y_test = train_test_split(X, y_fine, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# svm
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))





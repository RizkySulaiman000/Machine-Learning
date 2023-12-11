#Nama       : Rizky Sulaiman
#NIM        : 202110370311257

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Masukan dataset klasifikasi nya.
dataset = pd.read_csv('iris.csv')

# Pisahkan atribut nya antara x dan  y
X = dataset.drop('target_class', axis=1)
y = dataset['target_class']

# pisahkan antara data training dan testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# algoritma klasifikasi DecisionTree.
clf1 = DecisionTreeClassifier()

# Proses untuk membentuk model.
clf1.fit(X_train, y_train)

# Model yang dihasilkan dari tahap sebelumnya, model ada pada variabel "clf".
y_pred1 = clf1.predict(X_test)

# Pengujian model dengan testing data.
accuracy = clf1.score(X_test, y_test)
print("Accuracy:", accuracy)

# Menampilkan hasil prediksi, khususnya data testing.
print("Predicted classes for the test data:")
for i in range(len(X_test)):
    print("Instance", i, "belongs to class", y_pred1[i])

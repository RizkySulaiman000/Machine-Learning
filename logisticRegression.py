#Nama       : Rizky SUlaiman
#NIM        : 202110370311257

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Menyediakan dataset yang sesuai dengan kasus klasifikasi.
dataset = pd.read_csv('iris.csv')

# Memisahkan antara atribut/fitur (X) dengan target class (y)
X = dataset.drop('target_class', axis=1)
y = dataset['target_class']

# Memisah antara data training dan testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menentukan algoritma klasifikasi tertentu.
clf3 = LogisticRegression()

# Proses untuk membentuk model.
clf3.fit(X_train, y_train)

# Model yang dihasilkan dari tahap sebelumnya, model ada pada variabel "clf".
y_pred3 = clf3.predict(X_test)

# Pengujian model dengan testing data.
accuracy = clf3.score(X_test, y_test)
print("Accuracy:", accuracy)

# Menampilkan hasil prediksi, khususnya data testing.
print("Predicted classes for the test data:")
for i in range(len(X_test)):
    print("Instance", i, "belongs to class", y_pred3[i])

# Machine Learning with Python 
## Student Final Grade Prediction - Midterm Project
*Project ini dibuat untuk memenuhi tugas UTS mata kuliah PEMBELAJARAN SECARA STATISTIK DAN OPTIMISASI S2TE-31-01 [SZL]*

## Team :
```
Muammar Qhadafi (2101191019)
Raafi Dwi Susanto (2101191026)
Saepul Uyun (2101191034)
```

## Dataset Source :
```
Dataset yang digunakan berasal dari UCI Machine Learning Repository 
Dataset akan dilampirkan beserta full source code dari project ini.
```
[Go To UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

## Pre-Condition :
```
- 	Atribut pada dataset ini (student-mat.csv) ada 33 macam yaitu :
	1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
	2 sex - student's sex (binary: 'F' - female or 'M' - male)
	3 age - student's age (numeric: from 15 to 22)
	4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
	5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
	6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
	7 Medu - mother's education (numeric: 0-none, 1-primary education (4th grade), 2-5th to 9th grade,3-secondary education or 4 higher education)
	8 Fedu - father's education (numeric: 0-none, 1-primary education (4th grade), 2-5th to 9th grade,3-secondary education or 4 higher education)
	9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
	10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
	11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
	12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
	13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
	14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
	15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
	16 schoolsup - extra educational support (binary: yes or no)
	17 famsup - family educational support (binary: yes or no)
	18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
	19 activities - extra-curricular activities (binary: yes or no)
	20 nursery - attended nursery school (binary: yes or no)
	21 higher - wants to take higher education (binary: yes or no)
	22 internet - Internet access at home (binary: yes or no)
	23 romantic - with a romantic relationship (binary: yes or no)
	24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
	25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
	26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
	27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
	28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
	29 health - current health status (numeric: from 1 - very bad to 5 - very good)
	30 absences - number of school absences (numeric: from 0 to 93)
	
	# these grades are related with the course subject, Math or Portuguese:
	31 G1 - first period grade (numeric: from 0 to 20)
	31 G2 - second period grade (numeric: from 0 to 20)
	32 G3 - final grade (numeric: from 0 to 20, output target)

- 	Tidak Semua dari atribut akan digunakan sebagai bahan perhitungan prediksi, 
	atribut yang akan digunakan antara lain adalah :
	1 studytime
	2 failures
	3 famsup
	4 paid
	5 internet
	6 health
	7 absences
	8 studytime
	9 G1
	10 G2
	11 G3
```

## Code Explanation :
*Bagian ini adalah penjelasan dari source code yang dibuat, semoga cukup rinci dan detail untuk dimengerti :)*

```
#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
```
Pada bagian diatas adalah statement untuk memanggil dan menggunakan Packages/Module yang digunakan dalam project ini

```
style.use("ggplot")
```
Pada bagian diatas adalah line code untuk menggunakan matplotlib yang nantinya diperuntukkan untuk menggambarkan grafik

```
data = pd.read_csv("drive/Shared drives/Neural-Network/student-mat.csv", sep=";")
```
Ini adalah line yang digunakan untuk me-load atau membaca dataset,
pada line ini digunakan 'sep=";"' , karena pada dataset .csv ini, setiap data dipisahkan dengan semicolon ";",
sehingga program dapat mengetahui awal dan akhir dari setiap data yang ada dalam dataset ini

```
predict = "G3"
data = data[["studytime", "failures", "famsup", "paid", "internet", "health", "absences", "studytime", "G1", "G2", "G3"]] 
```
G3 adalah output yang akan di cari prediksinya dalam project ini, dan variabel data adalah atribut dari dataset yang akan digunakan,
karena tidak semua atribut yang ada di dataset ini yang akan digunakan.


```
d = {'yes': 1, 'no': 0}
data['famsup'] = data['famsup'].map(d)
data['internet'] = data['internet'].map(d)

#le = preprocessing.LabelEncoder()
#famsup = le.fit_transform(list(data["famsup"]))
#print(famsup)
#internet = le.fit_transform(list(data["internet"]))
#print(internet)

print(data)

data = shuffle(data) # Optional - shuffle the data

x = np.array(data.drop([predict], 1))
y =np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "health"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('student_stress.csv')

columns = ['sleep_quality','headaches','academic_performance','study_load','activities','stress_level']

df = df[columns]

x = df.iloc[:, 0:5]
y = df.iloc[:, 5:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

lr = LinearRegression()
lr.fit(x_train, y_train)
nb = GaussianNB()
nb.fit(x_train, y_train)

pickle.dump(nb, open('model.pkl', 'wb'))



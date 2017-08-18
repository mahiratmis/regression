import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style



df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'                     # column to be predicted
df.fillna(value=-99999, inplace=True)           # fill unavailable values with outliers
forecast_out = int(math.ceil(0.01 * len(df)))    # 1% of the data will be forecasted

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)     # X to the range of -1 and +1


# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# a1 = a[-2:]
# a2 = a[:-2]


X_lately = X[-forecast_out:]   # last forecast_out rows
X = X[:-forecast_out]          # all rows except last forecast_out
y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan
last_date = df.iloc
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day



import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Get the Data
# =============================================================================
car = pd.read_csv('car.csv')
# =============================================================================
# Discover the Data
# =============================================================================
car.head()

car.info()

car.describe()

car["class"].value_counts()

from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
car["class"]=labelencoder.fit_transform(car["class"])
car["buying"]=labelencoder.fit_transform(car["buying"])
car["maint"]=labelencoder.fit_transform(car["maint"])
car["doors"]=labelencoder.fit_transform(car["doors"])
car["persons"]=labelencoder.fit_transform(car["persons"])
car["lug_boot"]=labelencoder.fit_transform(car["lug_boot"])
car["safety"]=labelencoder.fit_transform(car["safety"])

car.hist(bins=50, figsize=(20,15))
plt.show()
# =============================================================================
# Visualization
# =============================================================================
car.plot(kind="scatter", x="buying", y="class", alpha=0.8)
car.plot(kind="scatter", x="persons", y="class", alpha=0.8)
car.plot(kind="scatter", x="safety", y="class", alpha=0.8)

from pandas.tools.plotting import scatter_matrix
attributes = ["buying", "maint", "doors", "class", "persons", "lug_boot", "safety"]
scatter_matrix(car[attributes], figsize=(20, 15), alpha=0.4)

corr_matrix = car.corr()
print(corr_matrix)

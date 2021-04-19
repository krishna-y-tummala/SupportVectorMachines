#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#Change Working Directory if required
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\SVM')

#Flower Images
from PIL import Image
import PIL
im1 = Image.open('C:\\Users\\User\\Documents\\Iris_setosa.jpg')
im1.show()
im1.save('setosa.jpg')

im2 = Image.open('C:\\Users\\User\\Documents\\Iris_versicolor.jpg')
im2.show()
im2.save('versicolor.jpg')

im3 = Image.open('C:\\Users\\User\\Documents\\Iris_virginica.jpg')
im3.show()
im3.save('virginica.jpg')

#LOAD DATASET

iris = sns.load_dataset('iris')

#EDA
print('\n',iris.head(),'\n')
print('\n',iris.describe(),'\n')
print('\n',iris.info(),'\n')

#pairplot
i1 = sns.pairplot(data=iris,hue='species',diag_kind='hist',palette='Set1')
i1.savefig('PairPlot.jpg')
plt.show()

#From pairplot, Setosa is most separable
#KDE plot of sepal_width vs sepal_length for Setosa species
i2 = sns.kdeplot(data=iris[iris['species'] == 'setosa'],x='sepal_width',y='sepal_length',cmap='plasma',fill=True)
plt.savefig('KDEPlot-Setosa.jpg')
plt.show()

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X =iris.drop('species',axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Support Vectors
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

#Evaluation
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix
print('\nSVC Confusion Matrix:','\n',confusion_matrix(y_test,predictions),'\n')
print('\nSVC Classification Report:','\n',classification_report(y_test,predictions),'\n')

sns.set_style('white')
plot_confusion_matrix(model,X_test,y_test)
plt.savefig('SVC Confusion Matrix.jpg')
plt.show()

#GRIDSEARCH(To optimize SVC parameters)
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[1,0.1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

GridSearch = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

GridSearch.fit(X_train,y_train)

print('\n','Best Estimator:','\n',GridSearch.best_estimator_,'\n')

preds=GridSearch.predict(X_test)
print('\nGrid Search Confusion Matrix:','\n',confusion_matrix(y_test,preds),'\n')
print('\nGrid Search Classification Report:','\n',classification_report(y_test,preds),'\n')

plot_confusion_matrix(GridSearch,X_test,y_test)
plt.savefig('Grid Search Confusion Matrix.jpg')
plt.show()

print('\nThe metrics are similar because the dataset is small. Grid Search is always recommended for an SVM\n')









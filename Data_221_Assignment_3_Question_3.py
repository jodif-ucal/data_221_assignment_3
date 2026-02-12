import pandas as pd
from sklearn.model_selection import train_test_split

kidney_disease_data = pd.read_csv("csv_and_txt_files/kidney_disease.csv")

feature_matrix = kidney_disease_data.drop("classification", axis=1)
label_column = kidney_disease_data["classification"]

features_train, features_test, classification_train, classification_test = train_test_split(
    feature_matrix, label_column, test_size=0.7
)

#We should not train and test a model on the same data as the model has already been fitted on
# that data. The model has already seen those data points before, so the predictions may be
# accurate, but it will not be a good reflection on its ability to predict labels
# for other data points

#The purpose of the testing set is to see how well our model can predict labels for other sets
# of data apart from our training set. We can make measurements on how accurate our model
# is at predicting labels on new data, and can take further action from there e.g. train our
# model on more new data so that it becomes more accurate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Data_221_Assignment_3_Question_3 as ckd_data
import pandas as pd

#Setting the different values of k beforehand
k_values = [1, 3, 5, 7, 9]
accuracy_of_models = []

for value_of_k in k_values:
    #Using the current value of k from the loop for the model
    knn_model = KNeighborsClassifier(n_neighbors=value_of_k)
    trained_knn_model = knn_model.fit(ckd_data.features_train, ckd_data.classification_train)
    predicted_classifications = trained_knn_model.predict(ckd_data.features_test)

    #Saving the model's performance for the table later
    accuracy_of_models.append({
        "Value of k": value_of_k,
        "Accuracy": accuracy_score(ckd_data.classification_test, predicted_classifications),
        "Precision score": precision_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd"),
        "Recall score": recall_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd"),
        "F1 score": f1_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd")
    })

#Saving results into a pandas DataFrame and printing the results
accuracy_of_models_table = pd.DataFrame(accuracy_of_models)
print(accuracy_of_models_table)

#Choosing different values of k seemingly does not change the accuracy of the model, however
# the precision score increases with higher values of k and the recall and f1 score decrease
# with higher values of k.

#Very small values of k may cause overfitting as not a lot of neighbours to a certain data point
# are being taken into account, so the model will struggle with classifying data points that are
# largely different to what it fitted on, which could explain why the lower values of k used here
# had better recall scores as it may have chosen too many true values, reflected by their worse
# precision scores

#Larger values of k on the other hand may cause underfitting as too many neighbours will be taken
# into account, which may cause it to fail to identify the patterns that points in the dataset
# follows. Taking the performance of the models we used here as an example, the models with
# higher values of k failed to identify the patterns between people with kidney disease and people
# without kidney disease, leading to its poor recall score.
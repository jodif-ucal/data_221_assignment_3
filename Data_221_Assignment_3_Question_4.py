import Data_221_Assignment_3_Question_3 as ckd_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

knn_model = KNeighborsClassifier(n_neighbors=5)
trained_knn_model = knn_model.fit(ckd_data.features_train, ckd_data.classification_train)

predicted_classifications = trained_knn_model.predict(ckd_data.features_test)

accuracy = accuracy_score(ckd_data.classification_test, predicted_classifications)
precision = precision_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd")
recall = recall_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd")
f1 = f1_score(ckd_data.classification_test, predicted_classifications, pos_label="ckd")
cm = confusion_matrix(ckd_data.classification_test, predicted_classifications)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print()
print("Confusion matrix:\n", cm)

#True positive: The people who the model correctly identified to have chronic kidney disease
#True negative: The people who the model correctly identified to not have chronic kidney disease
#False positive: The people who the model incorrectly identified to have chronic kidney disease (they do not)
#False negative: The people who the model incorrectly identified to not have chronic kidney disease (they do)

#Accuracy alone may not be enough because bad models can have a high accuracy too. Accuracy may tell us
# the percentage of its correct predictions, but a model which classifies every data point under one
# classification may also have a decent accuracy score too, depending on the difference between data points
# classified under one label and other data points under other labels.
#It is not enough to get most of its predictions right when it did not do a good job differentiating
# between different classifications based on a data point's features

#I believe that in the context of these data, the recall metric is the most important metric in measuring
# this model's performance. This is because, as missing out on a kidney disease can be very serious, it is
# very important that the model picks up on the most amount of people with kidney disease as it can, even if
# it falsely identifies people who do not have it, as the consequence for those people is a lot less severe.
#Recall measures the percentage of True responses a model picks up on, so it makes the most sense that this
# would be the most important metric.
#Precision is less important in this case, as the severity of having a bad precision score is not as dangerous
# as a bad recall score, and the f1 score evens the weight between the two, which we do not want as recall should
# be the singular most important metric here
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn import svm
from sklearn.svm import LinearSVC

def train_svm( train_sample_data, train_sample_label, test_data, test_label):

    # Fit a linear classifier
    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(train_sample_data, train_sample_label)
    y_pred = svclassifier.predict(test_data)

    f1 = f1_score(test_label,y_pred)
    roc = roc_auc_score(test_label,y_pred)

    return f1, roc

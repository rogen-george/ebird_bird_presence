import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn import svm
from sklearn.svm import LinearSVC


land_cover = pd.read_excel("readings_normalized_landcover.xlsx", usecols = "IS:JH")

marsh_wren = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['Marsh Wren'])
yellow_throat = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['Common Yellowthroat'])
red_winged = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['Red-winged Blackbird'])
western_meadowlark = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['Western Meadowlark'])

# Convert values less than threshold to 0 and greater than threshold to 1 for classification
def convert_to_classes ( data, threshold = 0.1 ):
    data[ data > threshold ] = 1
    data[ data <= threshold ] = 0
    return np.array(data)

threshold = 0
marsh_wren = convert_to_classes( marsh_wren, threshold )
yellow_throat = convert_to_classes( yellow_throat, threshold )
red_winged = convert_to_classes( red_winged, threshold )
western_meadowlark = convert_to_classes( western_meadowlark, threshold )

data = np.array( land_cover )

#print ("Marsh Wren", np.sum(marsh_wren) )
#print ( "Yellow Throat", np.sum(yellow_throat))
#print ( "Red Winged ", np.sum(red_winged))
label = western_meadowlark
print ( "Sum of birds" , np.sum(red_winged) )
# Delete columns where all readings are zero
#idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
#data = np.delete(data, idx, axis=1)

train_data, test_data, train_label, test_label = train_test_split( data, label, train_size = 0.5, test_size = 0.5, shuffle = True, stratify = label)

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(train_data, train_label)
y_pred = svclassifier.predict(test_data)

f1 = f1_score(test_label,y_pred)
roc = roc_auc_score(test_label,y_pred)

print ( "Print without anything ", f1, roc )

positive_train = []
negative_train = []
# Split train data into positive and negative to solve the problem of class imbalances
for i in range( len( train_data ) ):
    if train_label[i] == 1:
        positive_train.append( train_data[i] )
    else:
        negative_train.append( train_data[i] )

positive_train = np.array( positive_train )
negative_train = np.array( negative_train )

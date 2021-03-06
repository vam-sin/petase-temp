# Petase Temperature

Deep learning tool which uses the protein primary sequence for predicting metling points  

# Regression Results

## Alpha-Beta Hydrolases

### One-Hot Encoding

#### Artificial Neural Network

```
Results:
MSE: 443.39963 (BS 1)
MSE: 339.28094 (BS 2)
MSE: 270.67087 (BS 4)
MSE: 247.76332 (BS 8)
MSE: 243.97980 (BS 16)
```

#### Convolutional Neural Network

```
Results:
MSE: 418.02591 (BS 1)
MSE: 376.60104 (BS 2)
MSE: 213.95776 (BS 4) - RMSE: 14.627295 (Best)
MSE: 230.97935 (BS 8)
MSE: 220.33647 (BS 16)
```

### ProtBert

#### Artificial Neural Network

```
Results:
MSE: 394.21994 (BS 1)
MSE: 283.81451 (BS 2)
MSE: 73.92136 (BS 4) - Best
MSE: 113.67749 (BS 8) 
```

#### Convolutional Neural Network

```
Results: Normalized (Max: 100.0, Min: 20.0)
MSE: 0.01381 (BS 4)
MSE: 0.01095 (BS 8)
MSE: 0.01520 (BS 16)
```

## All sequences

### ProtBert

#### Convolutional Neural Network

```
Results: MAX/MIN: 120.0 0.0 (Normalized)
MSE: 0.01106 (BS 64) - Best
```

# Classification Results

## Alpha-Beta Hydrolases

### ProtBert

#### Convolutional Neural Networks

```
Classes: 
0: Less than 70 degrees
1: >= 70

Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        18
           1       1.00      1.00      1.00         4

    accuracy                           1.00        22
   macro avg       1.00      1.00      1.00        22
weighted avg       1.00      1.00      1.00        22

Confusion Matrix
[[18  0]
 [ 0  4]]
F1 Score
1.0
```

## All sequences

### ProtBert

#### Convolutional Neural Networks

```Without SMOTE
Classes: 
0: Less than 70 degrees
1: >= 70

Classification Report
              precision    recall  f1-score   support

           0       0.97      0.94      0.95      1117
           1       0.57      0.70      0.63       126

    accuracy                           0.92      1243
   macro avg       0.77      0.82      0.79      1243
weighted avg       0.92      0.92      0.92      1243

Confusion Matrix
[[1050   67]
 [  38   88]]
F1 Score
0.9193303887978407
```

```With SMOTE
Threshold 70
Accuracy: 0.97558
Classification Report: BS 32
              precision    recall  f1-score   support

           0       1.00      0.95      0.97      1121
           1       0.96      1.00      0.98      1131

    accuracy                           0.98      2252
   macro avg       0.98      0.98      0.98      2252
weighted avg       0.98      0.98      0.98      2252

Confusion Matrix
[[1069   52]
 [   3 1128]]
F1 Score
0.9755633329631406
```

# Is-PETase Prediction Results

```Ground Truth Values:
Classes (>=70): [0,0,0,0,0,1]
Temperatures: [49, 55, 56, 49, 58, 79]
```

```
Predictions Convolutional Neural Network + ProtBert + All Sequnces (SMOTE) + Classification Method:
[[1.0000000e+00 2.3519826e-08] - 0
 [1.0000000e+00 2.8369568e-08] - 0 
 [9.9999976e-01 2.1806292e-07] - 0
 [1.0000000e+00 4.0027592e-11] - 0
 [9.9999905e-01 9.4503071e-07] - 0
 [9.9142951e-01 8.5705305e-03]] - 0

Predictions Convolutional Neural Network + ProtBert + All Sequences (Not SMOTE) + Classification Method:
[[9.9961984e-01 3.8015953e-04] - 0
 [9.9972147e-01 2.7857331e-04] - 0
 [9.9971801e-01 2.8191364e-04] - 0
 [9.9975854e-01 2.4149123e-04] - 0
 [9.9953949e-01 4.6047330e-04] - 0
 [9.9649519e-01 3.5047936e-03]] - 0

Predictions Convolutional Neural Network + ProtBert + Alpha-Beta Hydrolases + Classification Method:
[[0.9630317  0.03696835] - 0
 [0.9674205  0.03257949] - 0
 [0.9740097  0.02599025] - 0
 [0.959992   0.04000795] - 0
 [0.96564263 0.03435737] - 0
 [0.61113536 0.38886467]] - 0

 Predictions Convolutional Neural Network + ProtBert + Alpha-Beta Hydrolases + Classification Method (Tuned to PETases):
 [[0.9797538  0.02024623] - 0
 [0.97991234 0.02008769] - 0
 [0.98267937 0.01732068] - 0
 [0.97580373 0.02419624] - 0
 [0.9801838  0.01981621] - 0
 [0.33680946 0.66319054]] - 1
```

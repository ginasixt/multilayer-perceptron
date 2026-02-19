# PyTorch MLP for Diabetes Prediction

This project implements a **Multilayer Perceptron (MLP)** using PyTorch to predict the likelihood of diabetes based on health indicators.

The model is trained on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

---
## What I Learned

- How to design and implement an MLP in PyTorch
- What each layer and activation function does
- Why non-linearity is essential in deep learning
- How weights + ReLU allow neurons to specialize and respond to meaningful input patterns
- How sigmoid activation in the output provides interpretable probabilities
- How the training works

## Understanding the Internal Logic and Structure of the MLP
### Input Layer
- 21 input features (e.g., BMI, age, etc.)
- Not a “real layer” in PyTorch — just the raw input data

### 1st Hidden Layer
- 64 neurons
- Each neuron receives all 21 inputs
- In total: 64 neurons × 21 weights + 64 biases

### 2nd Hidden Layer
- 32 neurons
- Each neuron receives up to 64 inputs from the previous layer

### Output Layer
- 1 neuron
- Uses the sigmoid activation function to output a probability
- Returns a value between 0–1 representing the probability of diabetes being present

#### General Flow Between Layers
In each layer, every single neuron receives all outputs from the previous layer (dense connection) \
Each input is multiplied by the trainable weight, and a bias is added \
The resulting value (weighted sum + bias) is passed through an activation function (e.g., ReLU and for the Output Layer Sigmoid) \

## How ReLU Enables Learning
The Rectified Linear Unit (ReLU) activation function is defined as:

$`
f(z) = \max(0, z)
`$ \

- Outputs $`0`$ if $` z \le 0 `$  → the neuron is inactive for that forward pass  
- Outputs $` z `$ if $` z > 0 `$ → the neuron becomes active and passes the value to the next layer  

For a neuron with pre-activation value

$`
z = w_1 x_1 + w_2 x_2 + \dots + b
`$ \

the neuron contributes to the next layer only if $` z > 0`$ \

ReLU does not explicitly encode logical rules such as  
“high BMI + no exercise + age > 60”,  
but it enables such patterns to emerge during training. The network adjusts its weights and biases so that only specific combinations of inputs cause the neuron to activate.

Without ReLU (or another non-linear activation function), even deep networks are only a single linear transformation, incapable of modeling complex relationships.

## Variance and the Role of Kaiming Initialization

ReLU changes the distribution of activations by setting all negative values to zero. 
If the inputs to a layer are approximately symmetric around zero, about half of the values become zero after applying ReLU. In expectation, this reduces the variance of the activations by roughly a factor of one half.

Like we have the following signal:

[−2,−1,0,1,2]

ReLU transforms it to:

[0,0,0,1,2], 

so half or the values bacome zero.

If this variance reduction accumulates across multiple layers this can cause a unstable variance throughout the layers, the signal can progressively shrink, leading to vanishing gradients. Conversely, poorly scaled weights can also cause activations to grow across layers, leading to exploding gradients. 

If the variance sinks activations become increasingly close to zero, the gradients computed during backpropagation also shrink, this causes that the network may still technically train, but learning becomes very slow, early layers learn almost nothing, and the network behaves much shallower, no need for a deep network then if some layers do nothing.

If the variance grows (exploding signal), activations become very large and gradients can grow exponentially during backpropagation, which leads to the exploding gradient problem which causes numerical instability (overflow, NaNs), extremely large parameter updates and diverging loss during training.

Therefore, it is desirable to keep the variance of activations approximately constant across layers.

Kaiming initialization addresses this issue by scaling the variance of the weights proportionally to:

$`
\frac{2}{n_{in}}
`$ \

where $` n_{in} `$  is the number of input units. The factor 2 compensates for the expected variance reduction introduced by ReLU, helping to maintain stable signal propagation during both forward and backward passes.

This is more important for deeper MLP, so I have not implemented it here, but still good to know :)


## How the training works
The model is further optimized in each epoch cycle (i.e., a complete iteration over the training data set). The goal is to change the model parameters (weights and biases) so that the prediction comes closer and closer to the actual label.
The log below shows how the binary cross-entropy loss decreased over 1000 training iterations,
indicating that the model is learning to better distinguish between diabetic and non-diabetic cases:

```text
Epoch 0, Loss: 0.7728
Epoch 100, Loss: 0.3241
Epoch 200, Loss: 0.3164
Epoch 300, Loss: 0.3142
Epoch 400, Loss: 0.3131
Epoch 500, Loss: 0.3124
Epoch 600, Loss: 0.3119
Epoch 700, Loss: 0.3115
Epoch 800, Loss: 0.3112
Epoch 900, Loss: 0.3110
```

### The process is divided into five steps:
Through this process, the model learns to adjust the influence of each input feature (e.g. BMI, age, smoking) to improve prediction.

#### 1. Forward Pass
```python
outputs = model(X_train)
```

The input matrix X_train (size: [n_samples, n_features]) is propagated through the network. \
Each layer performs: \
$`z = XW + b`$ \
where W is the weight matrix and b is the bias vector.\
A non-linear activation function (e.g., ReLU) is applied after each layer to allow complex relationships to emerge. \
The output layer applies a Sigmoid function to return probabilities between 0 and 1 for each patient.

#### 2. Compute Loss (Error)
```python
loss = criterion(outputs, y_train)
```
The Binary Cross Entropy Loss compares the predicted probabilities with the actual labels (y_train). \
The final loss is the mean over all samples in the batch

#### 3. Zero out Gradients
```python
optimizer.zero_grad()
```
PyTorch accumulates gradients by default.
This step clears previous gradients to avoid mixing them with the current batch.

#### 4. Backpropagation (Gradient Computation)
```python
loss.backward()
```
The `.backward()` function in PyTorch:
  - Automatically **traces the entire computation graph** from output back to input
  - Applies the **chain rule** to calculate the **gradient (slope)** for every parameter
  - These gradients are stored and will be used in the next step
 
These gradients indicate the direction to update each weight to reduce the loss.
PyTorch automatically calculates the gradient of the loss with respect to all parameters.

#### 5. Parameter Update
```python
optimizer.step()
```
After the backward pass, every parameter (weight or bias) has a **gradient**, the value that tells the model how much that parameter contributed to the error. \
For example a positive gradient means increasing this weight increases the error.
**SGD (Stochastic Gradient Descent)** uses these gradients to update each parameter slightly in the direction that reduces the loss. \
This adjustment is called a learning step.  
It’s controlled by a hyperparameter called the **learning rate** (often written as `lr`). 

**Example:** \
A weight currently has the value `0.5`, and its gradient is `+0.2`. \
If the learning rate is `0.01`, the new weight becomes: \
$`0,5 - 0,01 * 0,2 = 0,498`$ \
we decrease the weight slightly. So the optimizer nudges the weight in the right direction.  \
This is repeated for every parameter, every epoch.

## Evaluation
```text
Accuracy: 0.8679832860296437
Confusion Matrix:
 [[42846   893]
 [ 5805  1192]]
```
- True Negatives (no diabetes, predicted correctly): 42,846
- False Positives (no diabetes, predicted as yes): 893
- False Negatives (has diabetes, missed): 5,805
- True Positives (has diabetes, detected): 1,192

```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.98      0.93     43739
         1.0       0.57      0.17      0.26      6997

    accuracy                           0.87     50736
   macro avg       0.73      0.57      0.60     50736
weighted avg       0.84      0.87      0.84     50736
```
- Precision (1.0): 57% of predicted diabetics were actually diabetic
- Recall (1.0): Only 17% of actual diabetics were correctly detected
- F1-Score: shows poor diabetic detection performance

### Conclusion after first Deep Learning Model
The model still struggles with recall for diabetic cases. This could be improved by:
- Adding more layers or neurones, but with being aware for overfitting
- Adding more iterations, watching the Loss Log
- The dataset is unbalanced (more non-diabetics)
   - Adjusting pos_weight in the loss function
   - Oversampling of the minority class (SMOTE, RandomOverSampler)
   - Using focal loss instead of BCE to focus on difficult cases
- Using different optimizer like Adam and adjusting the learning rate
- weigt decay or dropout


## Improving my model

### 1. Adding Validation Data Set
I split data into training, validation, and test sets /
70% training, 15% validation, 15% test, to plot a diagramm and visualize potential overfitting during training. 
```text
Epoch 0, Loss: 0.6976
Epoch 100, Loss: 0.3207
Epoch 200, Loss: 0.3155
Epoch 300, Loss: 0.3135
Epoch 400, Loss: 0.3124
Epoch 500, Loss: 0.3117
Epoch 600, Loss: 0.3112
Epoch 700, Loss: 0.3108
Epoch 800, Loss: 0.3104
Epoch 900, Loss: 0.3099
[INFO] Plotting training and validation loss ...
```

<img width="500" height="300" alt="Training_vs_validation_Loss" src="https://github.com/user-attachments/assets/1eed5ee4-09df-41a7-87f0-d9c9ed1acc75" />

```text
=== Evaluation ===
Accuracy: 0.8676547881845895
Confusion Matrix:
 [[32107   733]
 [ 4303   909]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.98      0.93     32840
         1.0       0.55      0.17      0.27      5212

    accuracy                           0.87     38052
   macro avg       0.72      0.58      0.60     38052
weighted avg       0.84      0.87      0.84     38052
```

### 2. Balancing out my dataset with pos_weight
In the original dataset, the number of negative cases (non-diabetics) far exceeds the number of positive cases (diabetics). This class imbalance causes the model to prioritize the majority class during training, leading to high overall accuracy, but poor recall for diabetics. \
To address this, we introduc pos_weight, it increases the penalty for misclassifying the minority class (diabetics). It scales the positive (class 1) loss term to compensate for the imbalance, helping the model treat both classes more equally during training 

As a result, the model becomes more sensitive to the minority class, learning to identify diabetics more accurately  even at the cost of a slightly reduced overall accuracy. \
By introducing pos_weight, we change how the loss is computed. Now, the model is punished more harshly for failing to recognize diabetics, which naturally results in higher overall loss values during training:
```text
Epoch 0, Loss: 1.2060
Epoch 100, Loss: 0.8759
Epoch 200, Loss: 0.8639
Epoch 300, Loss: 0.8597
Epoch 400, Loss: 0.8571
Epoch 500, Loss: 0.8548
Epoch 600, Loss: 0.8522
Epoch 700, Loss: 0.8494
Epoch 800, Loss: 0.8462
Epoch 900, Loss: 0.8431
```

<img width="500" height="300" alt="with_pos_weight" src="https://github.com/user-attachments/assets/149b8bbe-84b0-4af5-9b43-4ec490402d92" />


```text
=== Evaluation ===
Accuracy: 0.7221433827394093
Confusion Matrix:
 [[23347  9493]
 [ 1080  4132]]
Classification Report:
               precision    recall  f1-score   support

         0.0       0.96      0.71      0.82     32840
         1.0       0.30      0.79      0.44      5212

    accuracy                           0.72     38052
   macro avg       0.63      0.75      0.63     38052
weighted avg       0.87      0.72      0.76     38052
```
We can now see in the confusion matrix that the model correctly identifies more diabetes patients. 
- False Negatives (has diabetes, missed)
  - before: 5,805
  - after using pos_weight: 1080
- True Positives (has diabetes, detected):
  - before: 1,192
  - after: 4132

However, it loses some accuracy for non-diabetics: 
- True Negatives (no diabetes, predicted correctly):
  - before: 42,846
  - after: 23347
- False Positives (no diabetes, predicted as yes):
  - before: 893
  - after: 9493
 
Although overall accuracy has decreased, recall for diabetics (1.0) has improved significantly (from 0.17 to 0.79), Even if more patients are incorrectly diagnosed as diabetic (the recall rate is lower), this is preferable to the opposite. Particularly in medical models, we prefer to risk false alarms as long as the probability of overlooking patients with diabetes is lower.

### Added stratify whwile splitting data
Because the data is not balanced we need to make sure, that the ratio of y (diabetes y/n) in the val, test and training data set, is the same as in the base data set. So 
With stratify=y, we now ensure that all splits (train/validation/test) have the same class distribution (diabetes y/n) as the original.
This is important for unbalanced datasets, the evaluation can looks better without this varaible, but is not representative.
With the variable our Evaluationa is indeed slightly worse:
```text
Accuracy: 0.7192788815305372
Confusion Matrix:
 [[23186  9564]
 [ 1118  4184]]
```
```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.95      0.71      0.81     32750
         1.0       0.30      0.79      0.44      5302

    accuracy                           0.72     38052
   macro avg       0.63      0.75      0.63     38052
weighted avg       0.86      0.72      0.76     38052
```
### Early Stopping
Early stop counters overfitting by stopping epochs, when the loss of the validation data is not improving anymore.
```text
Epoch 0, Loss: 1.1915, Val Loss 1.1857
Epoch 100, Loss: 0.8750, Val Loss 0.8759
Epoch 200, Loss: 0.8615, Val Loss 0.8662
Epoch 300, Loss: 0.8572, Val Loss 0.8648
Early stopping at epoch 328
```

```text
=== Evaluation ===
Accuracy: 0.7182014085987596
Confusion Matrix:
 [[23071  9679]
 [ 1044  4258]]
```
```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.96      0.70      0.81     32750
         1.0       0.31      0.80      0.44      5302

    accuracy                           0.72     38052
   macro avg       0.63      0.75      0.63     38052
weighted avg       0.87      0.72      0.76     38052
```
<img width="1000" height="600" alt="loss_plot" src="https://github.com/user-attachments/assets/88456fc1-07b3-482c-abc2-3bc99f91b5d2" />








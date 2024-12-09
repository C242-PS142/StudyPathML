
# StudyPathML
This project demonstrates a complete machine learning pipeline, from data preprocessing and feature scaling to model training, hyperparameter tuning, evaluation, and predictions. The objective is to train a neural network model that predicts multiple output variables based on input features.

## Data Preparation
### 1. Dataset
- Input Data: Features with shape (N, 50) (50 features per sample).
- Output Data: Multi-output labels with shape (N, 5) (5 target labels per sample).

### 2. Data Splitting
The dataset is split into:
- Training Set: 68%
- Validation Set: 17% (from training data)
- Test Set: 15%

Code snippet:

```bash
# Split data menjadi training+validation dan test set
X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.15, random_state=42)

# Split training+validation menjadi training dan validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

```

### 3. Feature Scaling
- Input Features: Scaled using MinMaxScaler.
- Output Labels: Scaled using MinMaxScaler.

```bash
# Scaling input
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scaling output
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

```

## Model Architecture
The neural network was designed based on the best hyperparameters obtained from Keras Tuner.
### 1. Architecture:
#### 1. Input Layer: 50 nodes (features).
#### 2. Hidden Layers with varying units and dropout rates:
- Layer 1: 416 units, 0.2 dropout
- Layer 2: 96 units, 0.2 dropout
- Layer 3: 256 units, 0.4 dropout
- Layer 4: 384 units, 0.1 dropout
- Layer 5: 224 units, 0.4 dropout 
- Layer 6: 288 units, 0.3 dropout
#### 3. Output Layer: 5 nodes (targets).

### 2. Activation Functions: 
ReLU for hidden layers, Sigmoid for the output layer.

### 3. Optimizer: 
Adam with a learning rate of 0.00025095748994520946.
### 4. Loss Function:
Mean Squared Error (MSE).

## Model Training
#### 1. Batch Size: 32
#### 2. Epochs: 50
#### 3. Training code : 
```bash
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'RootMeanSquaredError'])

history = model.fit(
    X_train_scaled, y_train_scaled, 
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=50, 
    batch_size=32,
    verbose=1

```

## Model Evaluation
#### 1. Test Data Metrics:
- R2 Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)

```bash
test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test Loss: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")

```

#### 2. Prediction Validation:
- The model's predictions are compared to the ground truth.
- Scatter Plot: Visualize predicted vs. true values for each target.

```bash
predict_and_validate(model, X_test_scaled, y_test_scaled)

```

## Prediction with New Data
To predict on new data:
#### 1. Preprocess using the trained scaler
```bash
new_data_scaled = scaler_X.transform(new_data)
predictions_scaled = model.predict(new_data_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

```

#### 2. Output predictions in the original scale.

## Saving the Model and Scalers
#### 1. Save the trained model:
```bash
model.save('Model_StudyPath.h5')

```

#### 2. Save the scalers:
```bash
joblib.dump(scaler_X, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

```

## Link
[Dataset](https://www.kaggle.com/datasets/tunguz/big-five-personality-test/data)

## Tools Used
- Programming Language: Python 3.12.0
- Libraries: TensorFlow, Keras, NumPy, pandas, scikit-learn, Matplotlib

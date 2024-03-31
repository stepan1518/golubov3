import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator

url = 'https://raw.githubusercontent.com/stepan1518/golubov3/main/classes.csv'
df = pd.read_csv(url)

df['Star color'] = df['Star color'].str.lower().str.strip()
df['Star color'] = df['Star color'].replace('blue-white', 'blue white')

df_encoded = pd.get_dummies(df, columns=['Star color', 'Spectral Class']).astype(int)

y = df_encoded['Star type']
df_encoded.drop('Star type', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.15, random_state=42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PyTorchClassifier(BaseEstimator):
    def __init__(self, input_size, hidden_size, num_classes, epochs=5000, batch_size=32, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        self.model = NeuralNetwork(self.input_size, self.hidden_size, self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            X = torch.FloatTensor(X)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()


# Преобразование данных в массивы NumPy перед использованием
# Convert pandas DataFrames to NumPy arrays
X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

# Определение параметров сетки для GridSearchCV
param_grid = {
    'hidden_size': [64, 128, 256],
    'epochs': [600, 700, 800],
    'batch_size': [32, 64, 128],
    'lr': [0.01]
}

# Создание экземпляра класса PyTorchClassifier
classifier = PyTorchClassifier(input_size=X_train.shape[1], hidden_size=64, num_classes=6)

# Использование GridSearchCV для настройки гиперпараметров
# grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_np, y_train_np)
#
# # Получение лучших параметров и лучшегоЫ результата
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# print("Best Parameters:", best_params)
# print("Best Score:", best_score)
#
# # Predict on the test set using the best model
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test_np)
# print(type(y_pred))
# # Calculate accuracy on the test set
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)

classifier.fit(X_train_np, y_train_np)
y_pred = classifier.predict(X_test_np)
print(f'Test accuracy : {accuracy_score(y_pred, y_test_np)}\n')

y_pred = classifier.predict(X_train_np)
print(f'Train accuracy : {accuracy_score(y_pred, y_train_np)}')
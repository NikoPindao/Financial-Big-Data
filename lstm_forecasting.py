import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Prétraitement des données pour les autres symboles
def preprocess_data_for_others(df, target_symbol):
    features = df[['open', 'high', 'low', 'close', 'volume']]
    target = df['market_regime']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Créer les séquences temporelles pour LSTM
    seq_length = 30
    sequences = []
    targets = []

    for i in range(len(features_scaled) - seq_length):
        sequences.append(features_scaled[i:i + seq_length])
        targets.append(target.iloc[i + seq_length])

    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets, scaler

# Modèle LSTM pour la prédiction
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])  # Utiliser le dernier état caché pour la prédiction
        return out

def train_and_evaluate(new_df, symbol_target):
    # Filtrer les données pour le symbole cible
    df_target = new_df[new_df['symbol'] == symbol_target]

    # Filtrer les données pour les autres symboles
    df_others = new_df[new_df['symbol'] != symbol_target]

    # Prétraiter les données
    X_others, y_others, scaler = preprocess_data_for_others(df_others, symbol_target)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_others, y_others, test_size=0.2, shuffle=False)

    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Initialiser le modèle
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = len(np.unique(y_others))

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Entraînement du modèle
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True):
            inputs, targets = batch
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64)):.4f}")

    # Évaluer le modèle sur les données de test
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Prédictions pour la crypto cible
    df_target_features = df_target[['open', 'high', 'low', 'close', 'volume']]
    df_target_scaled = scaler.transform(df_target_features)
    X_target = []

    for i in range(len(df_target_scaled) - 30):
        X_target.append(df_target_scaled[i:i + 30])
    X_target = np.array(X_target)

    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_market_regime = model(X_target_tensor)
        _, predicted_classes = torch.max(predicted_market_regime, 1)

    predicted_classes = predicted_classes.numpy()
    actual_values = df_target['market_regime'].iloc[30:].values

    # Calcul des métriques
    accuracy = accuracy_score(actual_values, predicted_classes)
    f1 = f1_score(actual_values, predicted_classes, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(actual_values, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(actual_values), yticklabels=np.unique(actual_values))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return model, scaler
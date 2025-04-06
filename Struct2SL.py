import os
import pickle
import numpy as np
import pandas as pd
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# Define the MLP model
class MLP(nn.Module):
    '''
    input_size: The dimension of the input features.
    hidden_sizes: A list of integers specifying the number of neurons in each hidden layer.
    output_size: The number of neurons in the output layer, usually corresponding to the number of categories of the task.
    dropout: Dropout ratio, used for regularization to prevent overfitting, the default value is 0.5.
    '''
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i < len(self.hidden_layers) - 1:  
                x = self.relu(x)  # Apply ReLU activation function
                x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def load_data(structure_folder, sequence_file, ppi_file, sl_pairs_file, sl_non_pairs_file):
    structure_features = {}
    for file in os.listdir(structure_folder):
        gene_name = os.path.splitext(file)[0]
        with open(os.path.join(structure_folder, file), 'r') as f:
            structure_features[gene_name] = np.array(f.readline().split(), dtype=np.float32)
            structure_features[gene_name] = structure_features[gene_name].astype(np.float32)

    with open(ppi_file, 'rb') as f:
        ppi_features = pickle.load(f)

    with open(sequence_file, 'rb') as f:
        sequence_features = pickle.load(f)

    # Flatten the feature data into a one-dimensional vector
    for gene in sequence_features:
        sequence_features[gene] = sequence_features[gene].flatten()

    ppi_features = {gene: np.array(features, dtype=np.float32) for gene, features in ppi_features.items()}

    sequence_features = {gene: np.array(features, dtype=np.float32) for gene, features in sequence_features.items()}

    sl_pairs_df = pd.read_csv(sl_pairs_file)
    sl_pairs = [(row['n1.name'], row['n2.name']) for index, row in sl_pairs_df.iterrows()]

    sl_non_pairs_df = pd.read_csv(sl_non_pairs_file)
    sl_non_pairs = [(row['n1.name'], row['n2.name']) for index, row in sl_non_pairs_df.iterrows()]

    return structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs

def preprocess_data(structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs, num_negative_samples=1):
    features = []
    labels = []
    pos_num = 0
    neg_num = 0


    valid_genes = set(structure_features.keys()) & set(sequence_features.keys()) & set(ppi_features.keys())
    known_sl_pairs = set(tuple(pair) for pair in sl_pairs)
    known_non_sl_pairs = set(tuple(pair) for pair in sl_non_pairs)

    # Processing positive samples
    for pair in sl_pairs:
        gene1, gene2 = pair
        if gene1 in valid_genes and gene2 in valid_genes:
            combined_features = np.concatenate([structure_features[gene1], sequence_features[gene1],
                                                ppi_features[gene1], structure_features[gene2],
                                                sequence_features[gene2], ppi_features[gene2]])
            features.append(combined_features)
            labels.append(int(1))
            pos_num += 1
    print("Positive samples added!")
    print("Number of positive samples:", pos_num)

    # Treat known non-SL pairs as negative samples
    for pair in sl_non_pairs:
        gene1, gene2 = pair
        if gene1 in valid_genes and gene2 in valid_genes:
            combined_features = np.concatenate([structure_features[gene1], sequence_features[gene1],
                                                ppi_features[gene1], structure_features[gene2],
                                                sequence_features[gene2], ppi_features[gene2]])
            features.append(combined_features)
            labels.append(int(0))
            neg_num += 1
    print("Known negative samples added!")
    print("Number of known negative samples:", neg_num)

    # If the number of negative samples is insufficient, add negative samples by random selection
    while neg_num < pos_num*num_negative_samples:
        gene1, gene2 = random.sample(valid_genes, 2)
        if (gene1, gene2) not in known_sl_pairs and (gene2, gene1) not in known_sl_pairs and \
           (gene1, gene2) not in known_non_sl_pairs and (gene2, gene1) not in known_non_sl_pairs:
            combined_features = np.concatenate([structure_features[gene1], sequence_features[gene1],
                                                ppi_features[gene1], structure_features[gene2],
                                                sequence_features[gene2], ppi_features[gene2]])
            features.append(combined_features)
            labels.append(int(0))
            neg_num += 1
            if neg_num % 100 == 0:  
                print(f"Generated {neg_num} negative samples.")
    print("Negative samples added!")
    print("Number of negative samples:", neg_num)

    features = np.array(features)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    print("Data standardization completed!")

    return np.array(features), np.array(labels)

def test_model(model, test_loader, threshold):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
        print('\033[95m' + 'Test Accuracy: {:.2f}%'.format(100 * correct / total) + '\033[0m')
        return correct / total

def evaluate_with_threshold(predictions, true_labels, threshold):
    binary_predictions = [1 if p >= threshold else 0 for p in predictions]
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, binary_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return specificity, accuracy, precision, recall

def evaluate_model(model, data_loader, threshold):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze(1).tolist())
            true_labels.extend(labels.tolist())
    auc = roc_auc_score(true_labels, predictions)
    aupr = average_precision_score(true_labels, predictions)
    f1 = f1_score(true_labels, [1 if p >= threshold else 0 for p in predictions])
    print(f'AUC: {auc:.5f}, AUPR: {aupr:.5f}, F1-score: {f1:.5f}')

def test_evaluate_model(model, data_loader, threshold):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze(1).tolist())
            true_labels.extend(labels.tolist())
    auc = roc_auc_score(true_labels, predictions)
    aupr = average_precision_score(true_labels, predictions)
    f1 = f1_score(true_labels, [1 if p >= threshold else 0 for p in predictions])
    specificity, accuracy, precision, recall = evaluate_with_threshold(predictions, true_labels, threshold)
    print('\033[95m' + f'AUC: {auc:.5f}, AUPR: {aupr:.5f}, F1-score: {f1:.5f}' + '\033[0m')
    return auc, aupr, f1, specificity, accuracy, precision, recall

if __name__ == "__main__":
    structure_folder = "./data/struct_features"
    sequence_file = "./data/sequence_features.npz"
    ppi_file = "./data/ppi_features.npz"
    sl_pairs_file = "./data/Human_SL_filtered.csv"
    sl_non_pairs_file = "./data/Human_nonSL.csv"

    structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs = load_data(structure_folder, sequence_file, ppi_file, sl_pairs_file, sl_non_pairs_file)
    features, labels = preprocess_data(structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs)
    # Check the length of features and labels
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Check the length of training data
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    nan_positions = np.where(np.isnan(features))
    indices = np.arange(len(X_train))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    test_accuracy_scores = []
    auc_scores = []
    aupr_scores = []
    f1_scores = []
    specificity_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = [] 
    threshold = 0.5 
    best_test_spe_loss = float('inf')
    #MLP
    for fold, (train_index, val_index) in enumerate(kfold.split(indices)):
        print(f"Start the {fold+1}th fold training!")
        train_features = X_train[train_index]
        val_features = X_train[val_index]
        train_labels = y_train[train_index]
        val_labels = y_train[val_index]
        print(f"Fold {fold+1} - Training features shape:", train_features.shape)
        print(f"Fold {fold+1} - Validation features shape:", val_features.shape)
        train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels))
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        hidden_sizes = [256, 128, 64]
        input_size = features.shape[1]
        output_size = 1
        model = MLP(input_size, hidden_sizes, output_size, dropout=0.5)
        pos_weight = torch.tensor([1.0])  
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        num_epochs = 10000
        best_val_loss = float('inf')
        best_model = None
        early_stop_patience = 15
        early_stop_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                l2_regularization = 0.0
                for param in model.parameters():
                    l2_regularization += torch.norm(param, 2)
                loss += l2_regularization * 0.001  
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            true_labels = []
            predictions = []
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    val_loss += loss.item()
                    predictions.extend(outputs.squeeze(1).tolist())
                    # predictions.extend(torch.sigmoid(outputs).squeeze(1).tolist())
                    true_labels.extend(labels.tolist())
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # print('\033[93m' + f'Fold {fold+1} - Learning Rate: {scheduler.get_lr()[0]}' + '\033[0m')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print('\033[91m' + f'Early stopping at epoch {epoch+1}' + '\033[0m')
                    break

            evaluate_model(model, val_loader, threshold)

        # Test the model performance on the test set
        model.load_state_dict(best_model)
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_accuracy = test_model(model, test_loader, threshold)
        auc, aupr, f1, specificity, accuracy, precision, recall = test_evaluate_model(model, test_loader, threshold)

        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                test_loss += loss.item()
        test_loss /= len(test_loader)

        test_accuracy_scores.append(test_accuracy)
        auc_scores.append(auc)
        aupr_scores.append(aupr)
        f1_scores.append(f1)
        specificity_scores.append(specificity)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

        results_folder = "./final_results/"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        results_modelname = os.path.join(results_folder, f"bestmodel.pt")
        if test_loss < best_test_spe_loss:
            best_test_spe_loss = test_loss
            torch.save(best_model, results_modelname)

    print("Results:")
    print("Test Accuracy: {:.5f} +/- {:.5f}".format(np.mean(test_accuracy_scores), np.std(test_accuracy_scores)))
    print("AUC: {:.5f} +/- {:.5f}".format(np.mean(auc_scores), np.std(auc_scores)))
    print("AUPR: {:.5f} +/- {:.5f}".format(np.mean(aupr_scores), np.std(aupr_scores)))
    print("F1-score: {:.5f} +/- {:.5f}".format(np.mean(f1_scores), np.std(f1_scores)))

    results_txtname = os.path.join(results_folder, f"results.txt")
    with open(results_txtname, "w") as f:
        f.write("Results:\n")
        f.write("Test Accuracy: {:.5f} +/- {:.5f}\n".format(np.mean(test_accuracy_scores), np.std(test_accuracy_scores)))
        f.write("AUC: {:.5f} +/- {:.5f}\n".format(np.mean(auc_scores), np.std(auc_scores)))
        f.write("AUPR: {:.5f} +/- {:.5f}\n".format(np.mean(aupr_scores), np.std(aupr_scores)))
        f.write("F1-score: {:.5f} +/- {:.5f}\n".format(np.mean(f1_scores), np.std(f1_scores)))

import os
import pickle
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Define the GAN generator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Generator, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# Define the GAN discriminator model
class Discriminator(nn.Module ):
    def __init__(self, input_size, hidden_sizes):
        super(Discriminator, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

# Define the GCN model
class GCN(nn.Module):
    '''
    input_size: The dimension of the input features.
    hidden_sizes: A list of integers specifying the number of neurons in each hidden layer.
    output_size: The number of neurons in the output layer, usually corresponding to the number of categories of the task.
    dropout: Dropout ratio, used for regularization to prevent overfitting, the default value is 0.5.
    '''
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(GCN, self).__init__()
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

    for pair in sl_pairs:
        gene1 = pair[0]
        gene2 = pair[1]
        if gene1 in structure_features and gene2 in structure_features \
                and gene1 in sequence_features and gene2 in sequence_features \
                and gene1 in ppi_features and gene2 in ppi_features:
            combined_features = np.concatenate([structure_features[gene1], sequence_features[gene1],
                                                ppi_features[gene1], structure_features[gene2],
                                                sequence_features[gene2], ppi_features[gene2]])
            features.append(combined_features)
            labels.append(int(1))
            pos_num += 1
    print("Positive samples added!")

    for pair in sl_non_pairs:
        gene1 = pair[0]
        gene2 = pair[1]
        if gene1 in structure_features and gene2 in structure_features \
                and gene1 in sequence_features and gene2 in sequence_features \
                and gene1 in ppi_features and gene2 in ppi_features:
            combined_features = np.concatenate([structure_features[gene1], sequence_features[gene1],
                                                ppi_features[gene1], structure_features[gene2],
                                                sequence_features[gene2], ppi_features[gene2]])
            features.append(combined_features)
            labels.append(int(0))
            neg_num += 1
            print(neg_num)
    print("Negative samples added!")
    print("Number of positive samples:",pos_num)
    print("Number of negative samples:",neg_num)

    sm = SMOTE(random_state=42)
    features, labels = sm.fit_resample(np.array(features), np.array(labels))

    features = np.array(features)
    # Data Standardization
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    np.save('mean.npy', mean)
    np.save('std.npy', std)
    print("Data standardization completed!")
    
    return np.array(features), np.array(labels)

def split_dataset(features, labels, test_size=0.2, val_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

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

def custom_loss(loss1, loss2, loss3, alpha):
    # Comprehensive loss function, which is a weighted combination of validation set loss and specificity loss
    total_loss = alpha * loss1 + alpha * loss2 + alpha * loss3 
    return total_loss

if __name__ == "__main__":
    structure_folder = "./final_data/struct_features"
    sequence_file = "./final_data/sequence_features.npz"
    ppi_file = "./final_data/ppi_features.npz"
    sl_pairs_file = "./final_data/Human_SL.csv"
    sl_non_pairs_file = "./final_data/Human_nonSL.csv"

    structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs = load_data(structure_folder, sequence_file, ppi_file, sl_pairs_file, sl_non_pairs_file)
    features, labels = preprocess_data(structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs)
    nan_positions = np.where(np.isnan(features))
    train_features, val_features, test_features, train_labels, val_labels, test_labels = split_dataset(features, labels)

    val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    generator_input_size = features.shape[1]
    discriminator_input_size = features.shape[1]
    hidden_sizes = [256, 128, 64]
    output_size = features.shape[1]

    # Initialize the generator and discriminator
    generator = Generator(generator_input_size, hidden_sizes, output_size)
    discriminator = Discriminator(discriminator_input_size, hidden_sizes)
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    num_epochs = 500
    batch_size = 64

    #GANs
    for epoch in range(num_epochs):
        for _ in range(1):  
            real_samples = torch.tensor(train_features[train_labels == 0][np.random.choice(len(train_features[train_labels == 0]), batch_size)])
            noise = torch.randn(batch_size, generator_input_size)
            fake_samples = generator(noise).detach()
            # Training the Discriminator
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator(real_samples)
            fake_outputs = discriminator(fake_samples)
            real_loss = criterion(real_outputs, torch.ones_like(real_outputs))
            fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()
            # Training the Generator
            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, generator_input_size)
            fake_samples = generator(noise)
            generator_loss = criterion(discriminator(fake_samples), torch.ones_like(fake_samples[:, :1])) 
            generator_loss.backward()
            generator_optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}')
    # Generate negative samples using the generator
    num_negative_samples = 1 * len(train_features[train_labels == 1]) - len(train_features[train_labels == 0])  #正负样本比例为1：1
    num_negative_samples = math.floor(num_negative_samples)
    noise = torch.randn(num_negative_samples, generator_input_size)
    generated_samples = generator(noise).detach().numpy()
    generated_samples = generated_samples
    train_features = np.vstack((train_features, generated_samples))
    train_labels = np.concatenate((train_labels, np.zeros(num_negative_samples)))

    test_accuracy_scores = []
    auc_scores = []
    aupr_scores = []
    f1_scores = []
    specificity_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    best_specificity = 0.0
    threshold = 0.5  
    #GCN
    for i in range(6):
        train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        hidden_sizes = [256, 128, 64]
        input_size = features.shape[1]
        output_size = 1
        model = GCN(input_size, hidden_sizes, output_size, dropout=0.5)
        pos_weight = torch.tensor([1.0])  
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

        num_epochs = 10000
        best_test_spe_loss = float('inf')
        best_val_spe_loss = float('inf')
        best_model = None
        early_stop_patience = 24
        early_stop_counter = 0

        print("Start the " + str(i+1) + "th training!")
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
                    true_labels.extend(labels.tolist())
            val_loss /= len(val_loader)
            auc = roc_auc_score(true_labels, predictions)
            aupr = average_precision_score(true_labels, predictions)
            f1 = f1_score(true_labels, [1 if p >= threshold else 0 for p in predictions])
            auc_loss = 1 - auc
            aupr_loss = 1 - aupr
            f1_loss = 1 - f1
            val_spe_loss = custom_loss(auc_loss, aupr_loss, f1_loss, alpha=0.33)
            val_spe_loss_tensor = torch.tensor(val_spe_loss, requires_grad=True)
            
            # Clear gradients, perform back propagation and parameter update
            optimizer.zero_grad()
            val_spe_loss_tensor.backward()
            optimizer.step()
            scheduler.step(val_spe_loss)
            print('\033[93m' + f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()}' + '\033[0m')
            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_spe_loss.item():.4f}, Validation Loss: {val_loss:.4f}')

            if val_spe_loss < best_val_spe_loss:
                best_val_spe_loss = val_spe_loss
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
        test_dataset = TensorDataset(torch.tensor(test_features), torch.tensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_accuracy = test_model(model, test_loader, threshold)
        auc, aupr, f1, specificity, accuracy, precision, recall = test_evaluate_model(model, test_loader, threshold)
        auc_loss = 1 - auc
        aupr_loss = 1 - aupr
        f1_loss = 1 - f1
        test_spe_loss = custom_loss(auc_loss, aupr_loss, f1_loss, alpha=0.33)

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
        if test_spe_loss < best_test_spe_loss:
            torch.save(best_model, results_modelname)

    print("Results:")
    print("Test Accuracy: {:.5f} +/- {:.5f}".format(np.mean(test_accuracy_scores), np.std(test_accuracy_scores)))
    print("AUC: {:.5f} +/- {:.5f}".format(np.mean(auc_scores), np.std(auc_scores)))
    print("AUPR: {:.5f} +/- {:.5f}".format(np.mean(aupr_scores), np.std(aupr_scores)))
    print("F1-score: {:.5f} +/- {:.5f}".format(np.mean(f1_scores), np.std(f1_scores)))
    print("specificity: {:.5f} +/- {:.5f}".format(np.mean(specificity_scores), np.std(specificity_scores)))





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

'''
特异度（Specificity）：特异度是真负例的样本中被正确预测为负类的比例，用于衡量模型对负类样本的区分能力。
准确率（Accuracy）：准确率是分类正确的样本数与总样本数的比例，用于衡量分类模型整体的预测准确程度。
精确率（Precision）：精确率是分类为正类的样本中真正为正类的比例，用于衡量模型在预测为正类时的准确程度。
召回率（Recall，也称为灵敏度或真正例率）：召回率是真正为正类的样本中被正确预测为正类的比例，用于衡量模型对正类样本的覆盖程度。
'''

# 定义生成器模型
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

# 定义判别器模型
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

# 定义GCN模型
class ComplexGCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(ComplexGCN, self).__init__()
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
            if i < len(self.hidden_layers) - 1:  # 最后一层不加Dropout
                x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# 加载数据
def load_data(structure_folder, sequence_file, ppi_file, sl_pairs_file, sl_non_pairs_file):
    structure_features = {}
    for file in os.listdir(structure_folder):
        gene_name = os.path.splitext(file)[0]
        with open(os.path.join(structure_folder, file), 'r') as f:
            structure_features[gene_name] = np.array(f.readline().split(), dtype=np.float32)
            # 显式转换数据类型为浮点型
            structure_features[gene_name] = structure_features[gene_name].astype(np.float32)

    with open(ppi_file, 'rb') as f:
        ppi_features = pickle.load(f)

    with open(sequence_file, 'rb') as f:
        sequence_features = pickle.load(f)

    # 遍历 sequence_features 中的每个基因，将特征数据展平为一维向量
    for gene in sequence_features:
        sequence_features[gene] = sequence_features[gene].flatten()

    # 显式转换 ppi_features 中的数据类型为浮点型
    ppi_features = {gene: np.array(features, dtype=np.float32) for gene, features in ppi_features.items()}

    # 显式转换 sequence_features 中的数据类型为浮点型
    sequence_features = {gene: np.array(features, dtype=np.float32) for gene, features in sequence_features.items()}

    sl_pairs_df = pd.read_csv(sl_pairs_file)
    sl_pairs = [(row['n1.name'], row['n2.name']) for index, row in sl_pairs_df.iterrows()]

    sl_non_pairs_df = pd.read_csv(sl_non_pairs_file)
    sl_non_pairs = [(row['x_name'], row['y_name']) for index, row in sl_non_pairs_df.iterrows()]

    return structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs

# 数据预处理
def preprocess_data(structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs, num_negative_samples=1):
    features = []
    labels = []
    pos_num = 0
    neg_num = 0

    # 添加正样本
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
            print("+++++++++++++++++++++正样本加1+++++++++++++++++++++++++")
            pos_num += 1
            print(pos_num)
    print("正样本添加完毕！")

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
            print("---------------------负样本加1--------------------------")
            neg_num += 1
            print(neg_num)
            if neg_num == pos_num/2:
                break
    print("负样本添加完毕！")

    print("正样本数量：",pos_num)
    print("负样本数量：",neg_num)

    # sm = SMOTE(random_state=42)
    # features, labels = sm.fit_resample(np.array(features), np.array(labels))

    # 将特征数据转换为 NumPy 数组
    features = np.array(features)

    # 数据标准化
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    print("完成数据标准化！")
    
    return np.array(features), np.array(labels)

# 划分数据集
def split_dataset(features, labels, test_size=0.2, val_size=0.1):
    # 先划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    # 再从训练集中划分验证集
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
    # 将概率转换为二分类预测结果（0或1），根据给定的阈值
    binary_predictions = [1 if p >= threshold else 0 for p in predictions]
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, binary_predictions)
    tn, fp, fn, tp = cm.ravel()
    # 计算特异性
    specificity = tn / (tn + fp)
    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # 计算召回率
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

    specificity, accuracy, precision, recall = evaluate_with_threshold(predictions, true_labels, threshold)

    print(f'AUC: {auc:.5f}, AUPR: {aupr:.5f}, F1-score: {f1:.5f}, specificity: {specificity:.5f}, Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}')

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

    print('\033[95m' + f'AUC: {auc:.5f}, AUPR: {aupr:.5f}, F1-score: {f1:.5f}, specificity: {specificity:.5f}, Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}' + '\033[0m')
    return auc, aupr, f1, specificity, accuracy, precision, recall

def custom_loss(loss1, loss2, loss3, alpha):
    # 综合损失函数，通过加权组合验证集损失和特异性损失
    total_loss = alpha * loss1 + alpha * loss2 + alpha * loss3
    return total_loss

if __name__ == "__main__":
    structure_folder = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/final_data/struct_features"
    sequence_file = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/final_data/sequence_features.npz"
    ppi_file = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/final_data/ppi_features.npz"
    sl_pairs_file = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/final_data/Human_SL.csv"
    sl_non_pairs_file = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/case_study/gene_nonsl_gene.csv"

    structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs = load_data(structure_folder, sequence_file, ppi_file, sl_pairs_file, sl_non_pairs_file)
    print("成功加载数据！")
    features, labels = preprocess_data(structure_features, sequence_features, ppi_features, sl_pairs, sl_non_pairs)
    # 检查features是否有无效值
    print(np.isnan(features).any())
    nan_positions = np.where(np.isnan(features))
    print("NaN 值的位置：", nan_positions)
    print("features形状:", features.shape)
    print("成功处理数据！")
    train_features, val_features, test_features, train_labels, val_labels, test_labels = split_dataset(features, labels)
    print("成功划分数据集！")

    # train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 统计训练集标签中 0 和 1 的数量
    num_zeros = np.sum(train_labels == 0)
    num_ones = np.sum(train_labels == 1)

    print("增强前训练集中标签为 0 的数量:", num_zeros)
    print("增强前训练集中标签为 1 的数量:", num_ones)

    # 统计训练集标签中 0 和 1 的数量
    num_zeros = np.sum(test_labels == 0)
    num_ones = np.sum(test_labels == 1)

    print("测试集中标签为 0 的数量:", num_zeros)
    print("测试集中标签为 1 的数量:", num_ones)

    val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("数据准备完毕！")

    test_accuracy_scores = []
    auc_scores = []
    aupr_scores = []
    f1_scores = []
    specificity_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []


    # print("开始训练！")

    # 在验证集上尝试不同的阈值，选择性能最好的阈值
    best_threshold = None
    best_specificity = 0.0
    thresholds = [0.7]  # 可以根据需要调整阈值范围

    results_folder = "/home/yuanrz/old/pdb_path/process_data_liyaxuan/final_results/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)


    for i in range(1):
        for threshold in thresholds:

            print("\033[34m当前阈值：", "\033[34m", threshold, "\033[0m")

            # 划分数据集
            train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # # 定义模型参数
            # hidden_sizes = [256, 128, 64]
            hidden_sizes = [128, 64]
            input_size = features.shape[1]
            output_size = 1

            # 初始化更复杂的模型
            model = ComplexGCN(input_size, hidden_sizes, output_size, dropout=0.5)
            # criterion = nn.BCELoss()


            # # 设置pos_weight为5，以平衡正负样本比例为4:1的情况
            # pos_weight = torch.tensor([0.31])

            # 创建BCEWithLogitsLoss损失函数对象，并传入pos_weight
            criterion = nn.BCEWithLogitsLoss()



            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
            # 添加学习率衰减策略
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

            num_epochs = 10000
            best_val_spe_loss = float('inf')
            best_test_spe_loss = float('inf')
            best_model = None
            early_stop_patience = 20
            early_stop_counter = 0

            print("开始第" + str(i+1) + "次训练！")

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    # 添加L2正则化
                    l2_regularization = 0.0
                    for param in model.parameters():
                        l2_regularization += torch.norm(param, 2)
                    loss += l2_regularization * 0.001  # 调整正则化系数
                    loss.backward()
                    # 使用梯度裁剪
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    running_loss += loss.item()

                # 在验证集上评估模型性能
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

                 # 计算特异性
                specificity, accuracy, precision, recall  = evaluate_with_threshold(predictions, true_labels, threshold)
                
                # 计算特异性损失
                specificity_loss = 1 - specificity
                accuracy_loss = 1 - accuracy
                precision_loss = 1 - precision
                recall_loss = 1 - recall
                
                auc = roc_auc_score(true_labels, predictions)
                aupr = average_precision_score(true_labels, predictions)
                f1 = f1_score(true_labels, [1 if p >= threshold else 0 for p in predictions])
                
                # 计算特异性损失
                auc_loss = 1 - auc
                aupr_loss = 1 - aupr
                f1_loss = 1 - f1
                
                # 计算综合损失
                # val_spe_loss = custom_loss(val_loss, specificity_loss, alpha=0.5)
                # val_spe_loss = custom_loss(accuracy_loss, specificity_loss, alpha=0.5)

                val_spe_loss = custom_loss(auc_loss, aupr_loss, f1_loss, alpha=0.33)

                # 将val_spe_loss转换为PyTorch张量
                val_spe_loss_tensor = torch.tensor(val_spe_loss, requires_grad=True)
                
                # 清零梯度，进行反向传播和参数更新
                optimizer.zero_grad()
                val_spe_loss_tensor.backward()
                optimizer.step()

                scheduler.step(val_spe_loss)
                print('\033[93m' + f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()}' + '\033[0m')

                # 打印训练信息
                print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_spe_loss.item():.4f}, Validation Loss: {val_loss:.4f}, Specificity Loss: {specificity_loss:.4f}')

                # 提前停止
                if val_spe_loss < best_val_spe_loss:
                    best_val_spe_loss = val_spe_loss
                    best_model = model.state_dict()
                    early_stop_counter = 0
                    print('\033[92m' + f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_spe_loss.item():.4f}, Validation Loss: {val_loss:.4f}, Specificity Loss: {specificity_loss:.4f}' + '\033[0m')
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print('\033[91m' + f'Early stopping at epoch {epoch+1}' + '\033[0m')
                        break

                evaluate_model(model, val_loader, threshold)

            # 加载表现最好的模型
            model.load_state_dict(best_model)

            # 统计训练集标签中 0 和 1 的数量
            num_zeros = np.sum(test_labels == 0)
            num_ones = np.sum(test_labels == 1)

            print("测试集中标签为 0 的数量:", num_zeros)
            print("测试集中标签为 1 的数量:", num_ones)

            # 在测试集上测试模型性能
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

            if test_spe_loss < best_test_spe_loss:
                results_modelname = os.path.join(results_folder, f"bestmodel_new.pt")
                # 保存最佳模型
                torch.save(best_model, results_modelname)
                print("最佳模型已保存！")

    print("结果:")
    print("best_threshold:",best_threshold)
    print("best_specificity:",best_specificity)
    print("Test Accuracy: {:.5f} +/- {:.5f}".format(np.mean(test_accuracy_scores), np.std(test_accuracy_scores)))
    print("AUC: {:.5f} +/- {:.5f}".format(np.mean(auc_scores), np.std(auc_scores)))
    print("AUPR: {:.5f} +/- {:.5f}".format(np.mean(aupr_scores), np.std(aupr_scores)))
    print("F1-score: {:.5f} +/- {:.5f}".format(np.mean(f1_scores), np.std(f1_scores)))
    print("specificity: {:.5f} +/- {:.5f}".format(np.mean(specificity_scores), np.std(specificity_scores)))

    # 构建当前阈值的结果文件名
    results_filename = os.path.join(results_folder, f"final_results_new.txt")

    with open(results_filename, "w") as f:
        f.write("结果:\n")
        f.write("当前阈值: {}\n".format(threshold))
        f.write("Test Accuracy: {:.5f} +/- {:.5f}".format(np.mean(test_accuracy_scores), np.std(test_accuracy_scores)))
        f.write("AUC: {:.5f} +/- {:.5f}".format(np.mean(auc_scores), np.std(auc_scores)))
        f.write("AUPR: {:.5f} +/- {:.5f}".format(np.mean(aupr_scores), np.std(aupr_scores)))
        f.write("F1-score: {:.5f} +/- {:.5f}".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("specificity: {:.5f} +/- {:.5f}".format(np.mean(specificity_scores), np.std(specificity_scores)))
        f.write("accuracy: {:.5f} +/- {:.5f}".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("precision: {:.5f} +/- {:.5f}".format(np.mean(precision_scores), np.std(precision_scores)))
        f.write("recall: {:.5f} +/- {:.5f}".format(np.mean(recall_scores), np.std(recall_scores)))




import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def road_vis(x,y):
    # t-SNE 可视化
    X_std = StandardScaler().fit_transform(x.cpu())
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_std)
    X_tsne_data = np.vstack((X_tsne.T, y)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['1st_Component', '2nd_Component', 'class'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='class', x='1st_Component', y='2nd_Component')
    plt.show()

def evaluation(model, feature_df, fold=100):
    x = model

    valid_labels = ['primary', 'secondary', 'tertiary', 'residential']
    id_dict = {idx: i for i, idx in enumerate(valid_labels)}
    y_df = feature_df.loc[feature_df['highway'].isin(valid_labels)]
    x = x[y_df['fid'].tolist()]
    y = torch.tensor(y_df['highway'].map(id_dict).tolist())

    split = x.shape[0] // fold

    device_flag = True

    y_preds = []
    y_trues = []
    for i in range(fold):
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

        x_train, x_eval = x[train_idx], x[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]

        model = Classifier(x.shape[1], len(valid_labels)).cuda()

        if device_flag:
            print('device: ', next(model.parameters()).device)
            device_flag = False

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        best_acc = 0.
        for e in range(1, 101):
            model.train()
            ce_loss = nn.CrossEntropyLoss()(model(x_train), y_train.cuda())

            opt.zero_grad()
            ce_loss.backward()
            opt.step()

            model.eval()
            logit = F.softmax(model(x_eval), -1).detach().cpu()
            y_pred = torch.argmax(logit, dim=1)
            acc = accuracy_score(y_eval.cpu(), y_pred, normalize=False)
            if acc > best_acc:
                best_acc = acc
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    macro_f1 = f1_score(y_trues, y_preds, average='macro')
    micro_f1 = f1_score(y_trues, y_preds, average='micro')
    print(f'road classification     | micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}')

    # road_vis(x, y)


# 5000 route / torch.cat([gps_road_joint_rep, route_road_joint_rep], dim=2)
# road classification     | micro F1: 0.4620, macro F1: 0.3053

# 5000 route / gps_road_joint_rep
# road classification     | micro F1: 0.4373, macro F1: 0.2789

# 5000 route / route_road_joint_rep
# road classification     | micro F1: 0.4338, macro F1: 0.2821

# 5000 route / torch.cat([gps_road_joint_rep.unsqueeze(2), route_road_joint_rep.unsqueeze(2)], dim=2)
# road classification     | micro F1: 0.7170, macro F1: 0.6864


# == Evaluation ===
# (100000, 20)  evaluation 3
# 6124
# road classification     | micro F1: 0.6688, macro F1: 0.6611


# (100000, 20) evaluation 2
# 6124
# road classification     | micro F1: 0.6663, macro F1: 0.6595

# (100000, 20) evaluation 1
# 6124
# road classification     | micro F1: 0.6633, macro F1: 0.6591
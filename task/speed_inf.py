import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

def speed_distribution_vis(pred, label):
    sns.set_style('white')
    sns.distplot(label, kde=True, hist=False, label='label')
    sns.distplot(pred, kde=True, hist=False, label='pred')
    plt.legend()
    plt.xlabel('speed')
    plt.ylabel('freq')
    plt.show()

def evaluation(model, feature_df, fold=5):
    x = model
    y = torch.tensor(feature_df['road_speed'].tolist())

    split = x.shape[0] // fold

    device_flag = True

    y_preds = []
    y_trues = []
    for i in range(fold):  #
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

        x_train, x_eval = x[train_idx], x[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]

        qt = QuantileTransformer(n_quantiles=5000, output_distribution='normal', random_state=0) # 标准正态化
        y_train = qt.fit_transform(y_train.reshape(-1, 1))
        y_train = torch.tensor(y_train.flatten(), dtype=torch.float)
        # y_train, mean, std = label_norm(y_train)  # 01标准化
        model = Regressor(x.shape[1]).cuda()

        if device_flag:
            print('device: ', next(model.parameters()).device)
            device_flag = False

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        best_mae = 1e9
        for e in range(1, 101):
            model.train()
            opt.zero_grad()
            loss = nn.MSELoss()(model(x_train), y_train.cuda())
            loss.backward()
            opt.step()

            model.eval()
            y_pred = model(x_eval).detach().cpu()
            # y_pred = pred_unnorm(y_pred, mean, std) # 01标准化
            y_pred = qt.inverse_transform(y_pred.reshape(-1, 1)) # # 标准正态化
            y_pred = torch.tensor(y_pred.flatten(), dtype=torch.float)
            mse = mean_squared_error(y_eval.cpu(), y_pred)
            if mse < best_mae:
                best_mae = mse
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds) ** 0.5
    print(f'travel speed estimation | MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    speed_distribution_vis(y_preds, y_trues)

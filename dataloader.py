#!/usr/bin/python
import os

import numpy as np
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
import json

def extract_weekday_and_minute_from_list(timestamp_list):
    weekday_list = []
    minute_list = []
    for timestamp in timestamp_list:
        dt = datetime.fromtimestamp(timestamp)
        weekday = dt.weekday() + 1
        minute = dt.hour * 60 + dt.minute + 1
        weekday_list.append(weekday)
        minute_list.append(minute)
    return torch.tensor(weekday_list).long(), torch.tensor(minute_list).long()

def prepare_gps_data(df,mat_padding_value,data_padding_value,max_len):
    """

    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        gps_data: (batch, gps_max_length, num_features)
        gps_assign_mat: (batch, gps_max_length)

    """

    # padding opath_list
    opath_list = [torch.tensor(opath_list, dtype=torch.float32) for opath_list in df['opath_list']]
    gps_assign_mat = rnn_utils.pad_sequence(opath_list, padding_value=mat_padding_value,
                                                               batch_first=True)
    # padding gps point data
    data_package = []
    for col in df.drop(columns='opath_list').columns:
        features = df[col].tolist()
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        features = rnn_utils.pad_sequence(features, padding_value=torch.nan, batch_first=True)
        features = features.unsqueeze(dim=2)
        data_package.append(features)

    gps_data = torch.cat(data_package, dim=2)


    # todo 临时处理的方式, 把时间戳那维特征置1
    gps_data[:, :, 0] = torch.ones_like(gps_data[:, :, 0])

    # 对除第一维特征进行标准化
    for i in range(1, gps_data.shape[2]):
        fea = gps_data[:, :, i]
        nozero_fea = torch.masked_select(fea, torch.isnan(fea).logical_not())  # 计算不为nan的值的fea的mean与std
        gps_data[:, :, i] = (gps_data[:, :, i] - torch.mean(nozero_fea)) / torch.std(nozero_fea)

    # 把因为数据没有前置节点因此无法计算，加速度等特征的nan置0
    gps_data = torch.where(torch.isnan(gps_data), torch.full_like(gps_data, data_padding_value), gps_data)

    return gps_data, gps_assign_mat

def prepare_route_data(df,mat_padding_value,data_padding_value,max_len):
    """

    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        route_data: (batch, route_max_length, num_features)
        route_assign_mat: (batch, route_max_length)

    """
    # padding capath_list
    cpath_list = [torch.tensor(cpath_list, dtype=torch.float32) for cpath_list in df['cpath_list']]
    route_assign_mat = rnn_utils.pad_sequence(cpath_list, padding_value=mat_padding_value,
                                                               batch_first=True)

    # padding route data
    weekday_route_list, minute_route_list = zip(
        *df['road_timestamp'].apply(extract_weekday_and_minute_from_list))

    weekday_route_list = [torch.tensor(weekday[:-1]).long() for weekday in weekday_route_list] # road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    minute_route_list = [torch.tensor(minute[:-1]).long() for minute in minute_route_list]# road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    weekday_data = rnn_utils.pad_sequence(weekday_route_list, padding_value=0, batch_first=True)
    minute_data = rnn_utils.pad_sequence(minute_route_list, padding_value=0, batch_first=True)

    # 分箱编码时间值
    # interval = []
    # df['road_interval'].apply(lambda row: interval.extend(row))
    # interval = np.array(interval)[~np.isnan(interval)]
    #
    # cuts = np.percentile(interval, [0, 2.5, 16, 50, 84, 97.5, 100])
    # cuts[0] = -1
    #
    # new_road_interval = []
    # for interval_list in df['road_interval']:
    #     new_interval_list = pd.cut(interval_list, cuts, labels=[1, 2, 3, 4, 5, 6])
    #     new_road_interval.append(torch.Tensor(new_interval_list).long())

    new_road_interval = []
    for interval_list in df['road_interval']:
        new_road_interval.append(torch.Tensor(interval_list).long())

    delta_data = rnn_utils.pad_sequence(new_road_interval, padding_value=-1, batch_first=True)

    route_data = torch.cat([weekday_data.unsqueeze(dim=2), minute_data.unsqueeze(dim=2), delta_data.unsqueeze(dim=2)], dim=-1)# (batch_size,max_len,2)

    # 填充nan
    route_data = torch.where(torch.isnan(route_data), torch.full_like(route_data, data_padding_value), route_data)

    return route_data, route_assign_mat

class StaticDataset(Dataset):
    def __init__(self, data, mat_padding_value, data_padding_value,gps_max_len,route_max_len):
        # 仅包含gps轨迹和route轨迹，route中包含路段的特征
        # 不包含路段过去n个时间戳的流量数据
        self.data = data
        gps_length = data['opath_list'].apply(lambda opath_list: self._split_duplicate_subseq(opath_list, data['route_length'].max())).tolist()
        self.gps_length = torch.tensor(gps_length, dtype=torch.int)
        gps_data, gps_assign_mat = prepare_gps_data(data[['opath_list', 'tm_list',\
                                                          'lng_list', 'lat_list',\
                                                          'speed', 'acceleration',\
                                                          'angle_delta', 'interval',\
                                                          'dist']], mat_padding_value, data_padding_value, gps_max_len)

        # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
        self.gps_data = gps_data # gps point shape = (num_samples,gps_max_length,num_features)

        # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
        self.gps_assign_mat = gps_assign_mat #  shape = (num_samples,gps_max_length)

        # todo 路段本身的属性特征怎么放进去
        route_data, route_assign_mat = prepare_route_data(data[['cpath_list', 'road_timestamp','road_interval']],\
                                                          mat_padding_value, data_padding_value, route_max_len)

        # route对应的信息，从road_interval生成，padding_value = 0
        self.route_data = route_data # shape = (num_samples,route_max_length,1)

        # 表示路段的序列信息，从cpath_list生成，padding_value = num_nodes
        self.route_assign_mat = route_assign_mat # shape = (num_samples,route_max_length)

    def _split_duplicate_subseq(self,opath_list,max_len):
        length_list = []
        subsequence = [opath_list[0]]
        for i in range(0, len(opath_list)-1):
            if opath_list[i] == opath_list[i+1]:
                subsequence.append(opath_list[i])
            else:
                length_list.append(len(subsequence))
                subsequence = [opath_list[i]]
        length_list.append(len(subsequence))
        return length_list + [0]*(max_len-len(length_list))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.gps_data[idx], self.gps_assign_mat[idx], self.route_data[idx], self.route_assign_mat[idx],
                self.gps_length[idx])

# class DynamicDataset_(Dataset):
#     def __init__(self, data, edge_features, traffic_flow_history):
#         # 仅包含gps轨迹和route轨迹，route中包含路段的特征
#         # 不包含路段过去n个时间戳的流量数据
#         self.data = data
#
#         # 特征矩阵和指示矩阵
#         # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
#         self.gps_data = # gps point shape = (num_samples,gps_max_length,features)
#
#         # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
#         self.gps_assign_mat = #  shape = (num_samples,gps_max_length)
#
#         # # route features
#         # # cpath_list
#         # # road_interval
#         # # 关联道路特征图有其他特征
#
#         # route对应的信息，从road_interval 和 edge_features生成，包含静态特征和动态特征，静态特征为路段特征，动态特征为历史流量信息，padding_value = 0
#         self.route_data = # shape = (num_samples,route_max_length,num_features)
#         self.route_assign_mat = # shape = (num_samples,route_max_length)
#
#         self.data_lst = []
#         self.data.groupby('pairid').apply(lambda group: self.data_lst.append(torch.tensor(np.array(group.iloc[:,:-2]))))
#         self.x_mbr = pad_sequence(self.data_lst, batch_first=True) # set_count*max_len*fea_dim
#         self.x_c = self.x_mbr
#         self.label_idxs = self.data.groupby('pairid').apply(lambda group:find_label_idx(group)).values # label=1 ndarray
#         self.lengths = self.data.groupby('pairid')['label'].count().values # ndarray
#
#     def __len__(self):
#         return len(self.data_lst)
#
#     def __getitem__(self, idx):
#         return (self.x_mbr[idx],self.x_c[idx],self.label_idxs[idx],self.lengths[idx])

def get_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)

    print(dataset.shape)
    assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    dataset['flag'] = pd.to_datetime(dataset['start_time'], unit='s').dt.day
    # 前13天作为训练集，第14天作为测试集，第15天作为验证集
    train_data, test_data, val_data =dataset[dataset['flag']<14], dataset[dataset['flag']==14], dataset[dataset['flag']==15]
    train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    # notice: 一般情况下

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    test_dataset = StaticDataset(test_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    val_dataset = StaticDataset(val_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader, val_loader, test_loader

def get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)

    print(dataset.shape)
    print(num_samples)
    assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    # 前13天作为训练集，第14天作为测试集，第15天作为验证集，已经提前分好
    train_data = dataset
    train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader

def random_mask(gps_assign_mat, route_assign_mat, gps_length, mask_token, mask_length=1, mask_prob=0.2):

    # mask route
    col_num = int(route_assign_mat.shape[1] / mask_length) + 1
    batch_size = route_assign_mat.shape[0]

    # mask的位置和padding的位置有重合，但整体mask概率无影响
    route_mask_pos = torch.empty(
        (batch_size, col_num),
        dtype=torch.float32,
        device=route_assign_mat.device).uniform_(0, 1) < mask_prob

    route_mask_pos = torch.stack(sum([[col]*mask_length for col in route_mask_pos.t()], []), dim=1)

    # 截断
    if route_mask_pos.shape[1] > route_assign_mat.shape[1]:
        route_mask_pos = route_mask_pos[:, :route_assign_mat.shape[1]]

    masked_route_assign_mat = route_assign_mat.clone()
    masked_route_assign_mat[route_mask_pos] = mask_token

    # mask gps
    masked_gps_assign_mat = gps_assign_mat.clone()
    gps_mask_pos = []
    for idx, row in enumerate(gps_assign_mat):
        route_mask = route_mask_pos[idx]
        length_list = gps_length[idx]
        unpad_mask_pos_list = sum([[mask] * length_list[_idx].item() for _idx, mask in enumerate(route_mask)], [])
        pad_mask_pos_list = unpad_mask_pos_list + [torch.tensor(False)] * (
                    gps_assign_mat.shape[1] - len(unpad_mask_pos_list))
        pad_mask_pos = torch.stack(pad_mask_pos_list)
        gps_mask_pos.append(pad_mask_pos)
    gps_mask_pos = torch.stack(gps_mask_pos, dim=0)
    masked_gps_assign_mat[gps_mask_pos] = mask_token
    # 获得每个gps点对应路段的长度

    return masked_route_assign_mat, masked_gps_assign_mat

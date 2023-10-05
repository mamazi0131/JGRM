import torch
from dataloader import prepare_gps_data, prepare_route_data
import math
from update_road_representation import MeanAggregator, WeightedMeanAggregator
import numpy as np

# 回归label标准化
def label_norm(y):
    std = torch.std(y)
    mean = torch.mean(y)
    y = (y-mean)/std
    return y, mean, std

# 预测结果反标准化
def pred_unnorm(pred,mean,std):
    pred = pred*std + mean
    return pred

# 获取数据中出现至少一次的路段
def get_road(df):
    road_list = []
    df['cpath_list'].apply(lambda row: road_list.extend(row))
    return list(set(road_list))

# 获取GPS轨迹中每个子路段上gps点的数量
def split_duplicate_subseq(opath_list, max_len):
    length_list = []
    subsequence = [opath_list[0]]
    for i in range(0, len(opath_list) - 1):
        if opath_list[i] == opath_list[i + 1]:
            subsequence.append(opath_list[i])
        else:
            length_list.append(len(subsequence))
            subsequence = [opath_list[i]]
    length_list.append(len(subsequence))
    return length_list + [0] * (max_len - len(length_list))

# 准备评估时使用的训练数据
def prepare_data(dataset, route_min_len, route_max_len, gps_min_len, gps_max_len):
    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    # prepare_gps_data 和 prepare_route_data 中对数据的特征进行了标准化

    # 处理route
    route_data, route_assign_mat = prepare_route_data(dataset[['cpath_list', 'road_timestamp', 'road_interval']], \
                                                      mat_padding_value, data_padding_value, route_max_len)
    masked_route_assign_mat = route_assign_mat  # 在evaluation时，关闭mask机制

    # 处理gps
    gps_length = dataset['opath_list'].apply(
        lambda opath_list: split_duplicate_subseq(opath_list, dataset['route_length'].max())).tolist()
    gps_length = torch.tensor(gps_length, dtype=torch.int)
    gps_data, gps_assign_mat = prepare_gps_data(dataset[['opath_list', 'tm_list', \
                                                      'lng_list', 'lat_list', \
                                                      'speed', 'acceleration', \
                                                      'angle_delta', 'interval', \
                                                      'dist']], mat_padding_value, data_padding_value, gps_max_len)
    masked_gps_assign_mat = gps_assign_mat # 在evaluation时，关闭mask机制

    return route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset

# 从全量的训练数据中采样部分数据，并保证出现的路段种类尽可能多
def fair_sampling(df,num_samples):
    road_list = []
    df['cpath_list'].apply(lambda row: road_list.extend(row))
    road_count = {}
    for rid in set(road_list):
        count = road_list.count(rid)
        road_count[rid] = 1/math.log(math.exp(1)+count)

    df['prob'] = df['cpath_list'].apply(lambda cpath_list: max([road_count[cpath] for cpath in cpath_list]))
    df = df.sample(num_samples, weights=df['prob'])
    return df

# 没见过的路段用邻域的embedding的平均值补充
def get_road_embedding1(road_list,gps_road_joint_rep,route_road_joint_rep,route_assign_mat,seq_model):
    # 处理在训练时被观测到的路段
    road_joint_rep = torch.cat([gps_road_joint_rep.unsqueeze(2), route_road_joint_rep.unsqueeze(2)], dim=2)
    road_embedding = torch.zeros((seq_model.node_embedding.weight.shape[0], road_joint_rep.shape[-1])).cuda()
    route_assign_mat = route_assign_mat[:road_joint_rep.shape[0]]
    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat==road_id)
        rep_list = [road_joint_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        road_rep = torch.mean(road_rep, dim=0)
        road_embedding[road_id] = road_rep

    # 处理在训练时被未被观测到的路段
    aggregator = MeanAggregator() # 邻居的均值
    road_rep = aggregator(road_embedding, seq_model.edge_index)
    indexes = torch.nonzero(torch.sum(road_embedding, dim=1) == 0)
    for index in indexes:
        road_embedding[index.item()] = road_rep[index.item()]

    return road_embedding.detach()

# 没见过的路段用邻域的embedding的加权平均值补充
def get_road_embedding2(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat, seq_model, city):
    # 处理在训练时被观测到的路段
    weight_path = '/data/mazp/dataset/JMTR/didi_{}/transition_prob_mat.npy'.format(city)
    road_joint_rep = torch.cat([gps_road_joint_rep.unsqueeze(2), route_road_joint_rep.unsqueeze(2)], dim=2)
    road_embedding = torch.zeros((seq_model.node_embedding.weight.shape[0], road_joint_rep.shape[-1])).cuda()
    route_assign_mat = route_assign_mat[:road_joint_rep.shape[0]]

    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat==road_id)
        rep_list = [road_joint_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        road_rep = torch.mean(road_rep, dim=0)
        road_embedding[road_id] = road_rep

    # 处理在训练时被未被观测到的路段
    trans_mat = np.load(weight_path)
    weights = []
    for i, j in seq_model.edge_index.transpose(0, 1):
        weights.append(trans_mat[i][j])
    weights = torch.tensor(weights).cuda()

    aggregator = WeightedMeanAggregator() # 邻居的加权均值
    road_rep = aggregator(road_embedding, seq_model.edge_index, weights)
    indexes = torch.nonzero(torch.sum(road_embedding, dim=1)==0)
    for index in indexes:
        road_embedding[index.item()] = road_rep[index.item()]

    return road_embedding.detach()

# 没见过的路段用路段作为序列长度为1的序列输入route_encoder得到
def get_road_embedding3(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat, seq_model):
    # 处理在训练时被观测到的路段
    road_joint_rep = torch.cat([gps_road_joint_rep.unsqueeze(2), route_road_joint_rep.unsqueeze(2)], dim=2)
    road_embedding = torch.zeros((seq_model.node_embedding.weight.shape[0], road_joint_rep.shape[-1])).cuda()
    route_assign_mat = route_assign_mat[:road_joint_rep.shape[0]]
    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat==road_id)
        rep_list = [road_joint_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        road_rep = torch.mean(road_rep, dim=0)
        road_embedding[road_id] = road_rep.cuda()

    # 处理在训练时被未被观测到的路段
    indexes = torch.nonzero(torch.sum(road_embedding, dim=1) == 0)
    unseen_route_assign_mat = indexes
    road_rep, _ = seq_model.encode_route(None, unseen_route_assign_mat, unseen_route_assign_mat)
    for i, index in enumerate(indexes):
        road_embedding[index.item()] = road_rep[i]

    return road_embedding.detach()

# 从观测到的轨迹生成路段的表示
# 输入的是完整的数据，包括路由与GPS
def get_road_emb_from_traj(seq_model, test_data, without_gps=False, batch_size=1024, update_road='mean', city='chengdu'):
    assert update_road in ['mean', 'weight', 'route'], 'update_road must be one of [\'mean\', \'weight\', \'route\']'

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = test_data

    if without_gps:
        gps_data = torch.zeros_like(gps_data) # 无实际意义，用于补位
        masked_gps_assign_mat = torch.zeros_like(masked_gps_assign_mat) # 无实际意义，用于补位
        gps_length = torch.ones_like(gps_length) # 无实际意义，用于补位

    # 都放到显存里面放不下，需要分batch处理
    max_len = dataset['route_length'].max()
    with torch.no_grad():
        gps_road_joint_rep_list, route_road_joint_rep_list = [], []
        for i in range(route_data.shape[0] // batch_size + 1):  # 最后不足batch_size的case不要了
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > route_data.shape[0]:
                end_idx = None
            batch_route_data = route_data[start_idx:end_idx].cuda()
            batch_masked_route_assign_mat = masked_route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_data = gps_data[start_idx:end_idx].cuda()
            batch_masked_gps_assign_mat = masked_gps_assign_mat[start_idx:end_idx].cuda()
            batch_route_assign_mat = route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_length = gps_length[start_idx:end_idx].cuda()
            _, _, _, _, gps_road_joint_rep, _, route_road_joint_rep, _ \
                = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
                            batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)
            del batch_route_data, batch_masked_route_assign_mat, batch_gps_data, batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length
            padding = torch.zeros(
                (
                gps_road_joint_rep.shape[0], max_len - gps_road_joint_rep.shape[1], gps_road_joint_rep.shape[2])).cuda()

            gps_road_joint_rep = torch.cat([gps_road_joint_rep, padding], dim=1).cpu()
            route_road_joint_rep = torch.cat([route_road_joint_rep, padding], dim=1).cpu()

            gps_road_joint_rep_list.append(gps_road_joint_rep)
            route_road_joint_rep_list.append(route_road_joint_rep)

            del gps_road_joint_rep, route_road_joint_rep
            # torch.cuda.empty_cache() # 清空显存

    gps_road_joint_rep = torch.cat(gps_road_joint_rep_list, dim=0).cpu()  # 注意gps_road_joint_rep_list中不能为空
    route_road_joint_rep = torch.cat(gps_road_joint_rep_list, dim=0).cpu()

    road_list = get_road(dataset)
    print('number of roads observed: {}'.format(len(road_list)))

    if update_road == 'mean':
        road_embedding = get_road_embedding1(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model)
    elif update_road == 'weight':
        road_embedding = get_road_embedding2(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model, city)
    elif update_road == 'route':
        road_embedding = get_road_embedding3(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model)
    return road_embedding

def get_road_emb_from_traj_nomode(seq_model, test_data, without_gps=False, batch_size=1024, update_road='mean', city='chengdu'):
    assert update_road in ['mean', 'weight', 'route'], 'update_road must be one of [\'mean\', \'weight\', \'route\']'

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = test_data

    if without_gps:
        gps_data = torch.zeros_like(gps_data) # 无实际意义，用于补位
        masked_gps_assign_mat = torch.zeros_like(masked_gps_assign_mat) # 无实际意义，用于补位
        gps_length = torch.ones_like(gps_length) # 无实际意义，用于补位

    # 都放到显存里面放不下，需要分batch处理
    max_len = dataset['route_length'].max()
    with torch.no_grad():
        gps_road_rep_list, route_road_rep_list = [], []
        for i in range(route_data.shape[0] // batch_size + 1):  # 最后不足batch_size的case不要了
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > route_data.shape[0]:
                end_idx = None
            batch_route_data = route_data[start_idx:end_idx].cuda()
            batch_masked_route_assign_mat = masked_route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_data = gps_data[start_idx:end_idx].cuda()
            batch_masked_gps_assign_mat = masked_gps_assign_mat[start_idx:end_idx].cuda()
            batch_route_assign_mat = route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_length = gps_length[start_idx:end_idx].cuda()
            gps_road_rep, _, route_road_rep, _ \
                = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
                            batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)
            del batch_route_data, batch_masked_route_assign_mat, batch_gps_data, batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length
            padding = torch.zeros(
                (
                gps_road_rep.shape[0], max_len - gps_road_rep.shape[1], gps_road_rep.shape[2])).cuda()

            gps_road_rep = torch.cat([gps_road_rep, padding], dim=1).cpu()
            route_road_rep = torch.cat([route_road_rep, padding], dim=1).cpu()

            gps_road_rep_list.append(gps_road_rep)
            route_road_rep_list.append(route_road_rep)

            del gps_road_rep, route_road_rep
            # torch.cuda.empty_cache() # 清空显存

    gps_road_rep = torch.cat(gps_road_rep_list, dim=0).cpu()  # 注意gps_road_joint_rep_list中不能为空
    route_road_rep = torch.cat(gps_road_rep_list, dim=0).cpu()

    road_list = get_road(dataset)
    print('number of roads observed: {}'.format(len(road_list)))

    if update_road == 'mean':
        road_embedding = get_road_embedding1(road_list, gps_road_rep, route_road_rep, route_assign_mat,
                                             seq_model)
    elif update_road == 'weight':
        road_embedding = get_road_embedding2(road_list, gps_road_rep, route_road_rep, route_assign_mat,
                                             seq_model, city)
    elif update_road == 'route':
        road_embedding = get_road_embedding3(road_list, gps_road_rep, route_road_rep, route_assign_mat,
                                             seq_model)
    return road_embedding

# 从观测到的轨迹生成轨迹的表示
# 输入的是完整的数据, 包括gps
def get_seq_emb_from_traj_withALLModel(seq_model, test_data, without_gps=False, batch_size=1024):

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, _ = test_data

    if without_gps:
        gps_data = torch.zeros_like(gps_data)  # 无实际意义，用于补位
        masked_gps_assign_mat = torch.zeros_like(masked_gps_assign_mat)  # 无实际意义，用于补位
        gps_length = torch.ones_like(gps_length)  # 无实际意义，用于补位

    # 都放到显存里面放不下，需要分batch处理
    with torch.no_grad():
        gps_traj_joint_rep_list, route_traj_joint_rep_list = [], []
        for i in range(route_data.shape[0] // batch_size + 1):  # 最后不足batch_size的case不要了
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > route_data.shape[0]:
                end_idx = None
            batch_route_data = route_data[start_idx:end_idx].cuda()
            batch_masked_route_assign_mat = masked_route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_data = gps_data[start_idx:end_idx].cuda()
            batch_masked_gps_assign_mat = masked_gps_assign_mat[start_idx:end_idx].cuda()
            batch_route_assign_mat = route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_length = gps_length[start_idx:end_idx].cuda()
            _, _, _, _, _, gps_traj_joint_rep, _, route_traj_joint_rep \
                = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
                            batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)
            del batch_route_data, batch_masked_route_assign_mat, batch_gps_data, batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length

            gps_traj_joint_rep_list.append(gps_traj_joint_rep)
            route_traj_joint_rep_list.append(route_traj_joint_rep)
            del gps_traj_joint_rep, route_traj_joint_rep
            # torch.cuda.empty_cache() # 清空显存

    gps_traj_joint_rep = torch.cat(gps_traj_joint_rep_list, dim=0)  # 注意gps_road_joint_rep_list中不能为空
    route_traj_joint_rep = torch.cat(route_traj_joint_rep_list, dim=0)

    if without_gps:
        traj_joint_rep = route_traj_joint_rep
    else:
        traj_joint_rep = torch.cat([gps_traj_joint_rep.unsqueeze(1), route_traj_joint_rep.unsqueeze(1)], dim=2)
        traj_joint_rep = torch.mean(traj_joint_rep, dim=1)

    return traj_joint_rep

def get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024):

    route_data, masked_route_assign_mat, _, _, route_assign_mat, _, _ = test_data

    # 都放到显存里面放不下，需要分batch处理
    with torch.no_grad():
        route_traj_rep_list = []
        for i in range(route_data.shape[0] // batch_size + 1):  # 最后不足batch_size的case不要了
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > route_data.shape[0]:
                end_idx = None
            batch_route_data = route_data[start_idx:end_idx].cuda()
            batch_masked_route_assign_mat = masked_route_assign_mat[start_idx:end_idx].cuda()
            batch_route_assign_mat = route_assign_mat[start_idx:end_idx].cuda()
            route_road_rep, route_traj_rep = seq_model.encode_route(batch_route_data, batch_route_assign_mat, batch_masked_route_assign_mat)
            del batch_route_data, batch_masked_route_assign_mat, batch_route_assign_mat
            route_traj_rep_list.append(route_traj_rep)
            del route_traj_rep
            # torch.cuda.empty_cache() # 清空显存

    route_traj_rep = torch.cat(route_traj_rep_list, dim=0)

    return route_traj_rep

# 从学到的节点的表示生成轨迹的表示
def get_seq_emb_from_node(node_embedding, route_assgin_mat, route_length, batch_size):
    route_assgin_mat = route_assgin_mat.long().numpy()
    all_seq_rep = []
    for i in range(route_assgin_mat.shape[0] // batch_size + 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > route_assgin_mat.shape[0]:
            end_idx = None
        data_batch = route_assgin_mat[start_idx:end_idx]
        length_batch = route_length[start_idx:end_idx]

        batch_seq_rep = []
        for i in range(len(data_batch)):
            seq_embedding = torch.zeros((1, node_embedding.shape[1]))
            for j in range(length_batch[i]):
                road_id = data_batch[i][j]
                road_embedding = node_embedding[road_id].cpu()
                seq_embedding += road_embedding
            seq_embedding = seq_embedding / length_batch[i]
            batch_seq_rep.append(seq_embedding)
        batch_seq_rep = torch.stack(batch_seq_rep, dim=0).squeeze(1)
        all_seq_rep .append(batch_seq_rep)

    all_seq_rep = torch.cat(all_seq_rep, dim=0).cuda()

    return all_seq_rep
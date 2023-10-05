import numpy as np
import faiss # https://zhuanlan.zhihu.com/p/320653340
import torch
import math
from evluation_utils import get_seq_emb_from_node,get_seq_emb_from_traj_withRouteOnly
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from shapely.geometry import Polygon
count = 0

def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)

    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index

# 随机选位置，随机替换
def query_prepare1(expid, cleaned_task_data, padding_id, num_queries, detour_rate=0.1):
    min_len, max_len = 10, 100
    num_samples = len(cleaned_task_data)

    def detour(rate): # 有1-rate的概率替换一个新的路段，原始设置0.9，基本就没换
        p = np.random.random_sample() # 产生[0,1)之间的随机数
        return np.random.randint(padding_id) if p > rate else padding_id

    np.random.seed(expid)
    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)

    for i in range(num_queries):
        row = cleaned_task_data.iloc[random_index[i]]
        detour_pos = np.random.choice(row['path_len'], int(row['path_len'] * detour_rate), replace=False) # 随机选几个位置
        path = [detour(0) if i in detour_pos else e for i, e in enumerate(row['cpath_list'])]
        q_arr[i, :row['path_len']] = np.array(path, dtype=np.int32)

    y = random_index[:num_queries]
    query_route_length = cleaned_task_data['path_len'].values[random_index[:num_queries]]

    return torch.LongTensor(q_arr), y, query_route_length

# 随机选位置，detour替换，垃圾版
def query_prepare2(expid, cleaned_task_data, padding_id, num_queries, indexes, detour_rate=0.1):
    min_len, max_len = 10, 100
    num_samples = len(cleaned_task_data)

    def detour(rate, road_id, indexes): # 有1-rate的概率替换一个新的路段，原始设置0.9，基本就没换
        p = np.random.random_sample() # 产生[0,1)之间的随机数
        return indexes[road_id][np.random.randint(indexes.shape[1])] if p > rate else padding_id

    np.random.seed(expid)
    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)

    for i in range(num_queries):
        row = cleaned_task_data.iloc[random_index[i]]
        detour_pos = np.random.choice(row['path_len'], int(row['path_len'] * detour_rate), replace=False) # 随机选几个位置
        path = [detour(0, e, indexes) if i in detour_pos else e for i, e in enumerate(row['cpath_list'])]
        q_arr[i, :row['path_len']] = np.array(path, dtype=np.int32)

    y = random_index[:num_queries]
    query_route_length = cleaned_task_data['path_len'].values[random_index[:num_queries]]

    return torch.LongTensor(q_arr), y, query_route_length

# 随机选位置，detour替换，单路段替换版
def query_prepare3(expid, cleaned_task_data, padding_id, num_queries, detour_base, detour_rate=0.1):
    np.random.seed(expid)
    min_len, max_len = 10, 100
    num_samples = len(cleaned_task_data)

    def detour(rate, path, detour_pos):
        new_path = []
        for idx in range(len(path)//2):
            road_id_pre = path[2 * idx]
            road_id_next = path[2 * idx + 1]
            if 2*idx+1 in detour_pos:
                detours = detour_base[(road_id_pre, road_id_next)]
                if len(detours)!=0:
                    pos = np.random.choice(len(detours), 1, replace=False)[0]
                    detour_anchor = detours[pos]
                else:
                    global count
                    count += 1
                    detour_anchor = [road_id_pre, road_id_next]

                p = np.random.random_sample() # 产生[0,1)之间的随机数
                if p > rate:
                    new_path.extend(detour_anchor)
                else:
                    new_path.extend([padding_id, padding_id])
            else:
                new_path.extend([road_id_pre, road_id_next])
        if len(path)%2!=0:
            new_path.extend([path[-1]])
        if len(new_path) > max_len:
            new_path = path
        return new_path

    np.random.seed(expid)
    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)

    query_route_length = []
    for i in range(num_queries):
        row = cleaned_task_data.iloc[random_index[i]]
        sample_pos = list(range(row['route_length']))[1::2] # 偶数位处理
        sample_num = int(len(sample_pos) * detour_rate)+1
        detour_pos = np.random.choice(sample_pos, sample_num, replace=False) # 随机选几个位置
        path = detour(0, row['cpath_list'], detour_pos)
        # print(detour_pos, len(row['cpath_list']) - len(path))
        # if len(row['cpath_list']) == len(path):
        #     global count
        #     count = count+1
        query_route_length.append(len(path))
        q_arr[i, :len(path)] = np.array(path, dtype=np.int32)

    y = random_index[:num_queries]

    return torch.LongTensor(q_arr), y, np.array(query_route_length)

# 带时间的prepare3
def query_prepare4(expid, cleaned_task_data, padding_id, num_queries, detour_base, feature_df, detour_rate=0.1):

    feature_df['road_speed'][feature_df['road_speed'] < 1] = feature_df['road_speed'].mean()
    feature_df['road_travel_time'] = (feature_df['length'] / feature_df['road_speed']).astype(int)
    history_travel_time_dict = {}
    for _, row in feature_df.iterrows():
        history_travel_time_dict[row['fid']] = row['road_travel_time']

    min_len, max_len = 10, 100
    num_samples = len(cleaned_task_data)

    def detour(rate, path, tm_list, detour_pos, history_travel_time_dict):
        # 开始和结束的位置不变
        start_time = tm_list[0]
        new_path = []
        for idx in range(len(path)//2):
            road_id_pre = path[2 * idx]
            road_id_next = path[2 * idx + 1]
            if 2*idx+1 in detour_pos:
                detours = detour_base[(road_id_pre, road_id_next)]
                if len(detours)!=0:
                    pos = np.random.choice(len(detours), 1, replace=False)[0]
                    detour_anchor = detours[pos]
                else:
                    global count
                    count += 1
                    detour_anchor = [road_id_pre, road_id_next]

                p = np.random.random_sample() # 产生[0,1)之间的随机数
                if p > rate:
                    new_path.extend(detour_anchor)
                else:
                    new_path.extend([padding_id, padding_id])
            else:
                new_path.extend([road_id_pre, road_id_next])
        if len(path)%2!=0:
            new_path.extend([path[-1]])
        if len(new_path) > max_len:
            new_path = path

        # 产生tm_list
        cur_travel_time_dict = {}
        for i in range(len(path)-1):
            cur_travel_time_dict[path[i]] = tm_list[i+1] - tm_list[i]

        tm_list = [start_time]
        for road in new_path:
            shift_ratio = np.random.uniform(-0.15, 0.15)
            if road in cur_travel_time_dict.keys():
                tm_list.append(tm_list[-1] + cur_travel_time_dict[road] * (1 + shift_ratio))
            else:
                tm_list.append(tm_list[-1] + history_travel_time_dict[road] * (1 + shift_ratio))

        return new_path, tm_list

    np.random.seed(expid)
    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)
    week_arr = np.full([num_queries, max_len], 0, dtype=np.int32)
    minute_arr = np.full([num_queries, max_len], 0, dtype=np.int32)

    query_route_length = []
    for i in range(num_queries):
        row = cleaned_task_data.iloc[random_index[i]]
        sample_pos = list(range(row['route_length']))[1::2] # 只在偶数位处理
        sample_num = int(len(sample_pos) * detour_rate)+1
        detour_pos = np.random.choice(sample_pos, sample_num, replace=False) # 随机选几个位置
        path, tm_list = detour(0, row['cpath_list'], row['road_timestamp'], detour_pos, history_travel_time_dict)

        week_list = []
        minute_list = []
        for tm in tm_list[0:-1]:
            dt = datetime.fromtimestamp(tm)
            weekday = dt.weekday() + 1
            minute = dt.hour * 60 + dt.minute + 1
            week_list.append(weekday)
            minute_list.append(minute)

        query_route_length.append(len(path))
        q_arr[i, :len(path)] = np.array(path, dtype=np.int32)
        week_arr[i, :len(path)] = np.array(week_list, dtype=np.int32)
        minute_arr[i, :len(path)] = np.array(minute_list, dtype=np.int32)

    y = random_index[:num_queries]
    tm_arr = torch.cat([torch.LongTensor(week_arr).unsqueeze(dim=2), torch.LongTensor(minute_arr).unsqueeze(dim=2), torch.zeros_like(torch.LongTensor(minute_arr).unsqueeze(dim=2)).long()], dim=-1)

    return torch.LongTensor(q_arr), y, np.array(query_route_length), tm_arr

# 按照总长的detour_rate选子序列进行替换
# def query_prepare5(expid, cleaned_task_data, padding_id, num_queries, trans_mat, feature_df, detour_rate=0.1):
#     np.random.seed(expid)
#
#     feature_df['road_speed'][feature_df['road_speed'] < 1] = feature_df['road_speed'].mean()
#     feature_df['road_travel_time'] = (feature_df['length'] / feature_df['road_speed']).astype(int)
#     history_travel_time_dict = {}
#     for _, row in feature_df.iterrows():
#         history_travel_time_dict[row['fid']] = row['road_travel_time']
#
#     min_len, max_len = 10, 100
#     num_samples = len(cleaned_task_data)
#
#     # def dfs_path(trans_mat, start, end, origin_path, origin_travel_time, history_travel_time_dict):
#     #     stack = [(start, [start])]
#     #     paths = []
#     #     while stack:
#     #         (vertex, path) = stack.pop()
#     #
#     #         if vertex == end:  # 如果到达终点，则将路径添加到结果列表中
#     #             paths.append(path)
#     #             travel_time = np.sum([history_travel_time_dict[road] for road in path])
#     #             if travel_time > origin_travel_time and len(path) <= int(1 / 3 * len(origin_path)) + 1:
#     #                 return path
#     #
#     #         if len(path) - 1 == 10:  # 如果长度到达最大长度但不符合标准则去除
#     #             continue
#     #
#     #         for neighbor in torch.nonzero(trans_mat[vertex] != 0).reshape(-1, ).numpy().tolist():
#     #             if neighbor not in path:  # 如果邻居节点尚未访问过，则将其添加到路径中，并将其压入栈中
#     #                 stack.append((neighbor, path + [neighbor]))
#     #     if len(paths) == 0:
#     #         return None
#     #     return paths[-1]
#
#     def dfs_path(trans_mat, start, end, detour_path, origin_path, origin_travel_time, history_travel_time_dict):
#         stack = [(start, [start])]
#         paths = []
#         while stack:
#             (vertex, path) = stack.pop()
#
#             if vertex == end and path != detour_path:  # 如果到达终点，则将路径添加到结果列表中
#                 paths.append(path)
#                 if len(path) <= int(1 / 3 * len(origin_path)) + 1:
#                     return path
#
#             if len(path) - 1 == 10:  # 如果长度到达最大长度但不符合标准则去除
#                 continue
#
#             for neighbor in torch.nonzero(trans_mat[vertex] != 0).reshape(-1, ).numpy().tolist():
#                 if neighbor not in path:  # 如果邻居节点尚未访问过，则将其添加到路径中，并将其压入栈中
#                     stack.append((neighbor, path + [neighbor]))
#         if len(paths) == 0:
#             return None
#         return paths[-1]
#
#     def detour(replace_rate, path, tm_list, detour_path, start_pos ,trans_mat, detour_travel_time, history_travel_time_dict):
#         # 需要重建detour_base
#         # 开始和结束的位置不变
#         start_time = tm_list[0]
#         detour_anchor = dfs_path(trans_mat, detour_path[0], detour_path[-1], detour_path, path, detour_travel_time, history_travel_time_dict)
#         if detour_anchor is None:
#             detour_anchor = detour_path
#         end_pos = start_pos + len(detour_path)
#         pre_path = path[:start_pos]
#         next_path = path[end_pos:]
#         p = np.random.random_sample()  # 产生[0,1)之间的随机数
#         if p > replace_rate:
#             new_path = pre_path + detour_anchor + next_path
#         else:
#             new_path = pre_path + [padding_id]*len(detour_path) + next_path
#
#         # 产生tm_list
#         cur_travel_time_dict = {}
#         for i in range(len(path)-1):
#             cur_travel_time_dict[path[i]] = tm_list[i+1] - tm_list[i]
#
#         tm_list = [start_time]
#         for road in new_path:
#             shift_ratio = np.random.uniform(-0.1, 0.1)
#             if road in cur_travel_time_dict.keys():
#                 tm_list.append(tm_list[-1] + cur_travel_time_dict[road] * (1 + shift_ratio))
#             else:
#                 tm_list.append(tm_list[-1] + history_travel_time_dict[road] * (1 + shift_ratio))
#         return new_path, tm_list
#
#     random_index = np.random.permutation(num_samples)
#     q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)
#     week_arr = np.full([num_queries, max_len], 0, dtype=np.int32)
#     minute_arr = np.full([num_queries, max_len], 0, dtype=np.int32)
#
#     query_route_length = []
#     for i in tqdm(range(num_queries)):
#         row = cleaned_task_data.iloc[random_index[i]]
#         sample_len = int(row['route_length'] * detour_rate) + 1
#         start_pos = np.random.randint(1, row['route_length'] - sample_len) # OD不参与处理
#         detour_path = row['cpath_list'][start_pos:start_pos+sample_len]
#         detour_travel_time = row['road_timestamp'][start_pos+sample_len-1] - row['road_timestamp'][start_pos]
#         path, tm_list = detour(0, row['cpath_list'], row['road_timestamp'], detour_path, start_pos, trans_mat, detour_travel_time, history_travel_time_dict)
#         week_list = []
#         minute_list = []
#         for tm in tm_list[0:-1]:
#             dt = datetime.fromtimestamp(tm)
#             weekday = dt.weekday() + 1
#             minute = dt.hour * 60 + dt.minute + 1
#             week_list.append(weekday)
#             minute_list.append(minute)
#
#         query_route_length.append(len(path))
#         q_arr[i, :len(path)] = np.array(path, dtype=np.int32)
#         week_arr[i, :len(path)] = np.array(week_list, dtype=np.int32)
#         minute_arr[i, :len(path)] = np.array(minute_list, dtype=np.int32)
#
#     y = random_index[:num_queries]
#     tm_arr = torch.cat([torch.LongTensor(week_arr).unsqueeze(dim=2), torch.LongTensor(minute_arr).unsqueeze(dim=2), torch.zeros_like(torch.LongTensor(minute_arr).unsqueeze(dim=2)).long()], dim=-1)
#
#     return torch.LongTensor(q_arr), y, np.array(query_route_length), tm_arr



# 不带时间的

def query_prepare5(expid, cleaned_task_data, padding_id, num_queries, trans_mat, feature_df, geometry_df, detour_rate=0.1):
    np.random.seed(expid)

    feature_df['road_speed'][feature_df['road_speed'] < 1] = feature_df['road_speed'].mean()
    feature_df['road_travel_time'] = (feature_df['length'] / feature_df['road_speed']).astype(int)
    history_travel_time_dict = {}
    road_length_dict = {}
    for _, row in feature_df.iterrows():
        history_travel_time_dict[row['fid']] = row['road_travel_time']
        road_length_dict[row['fid']] = row['length']

    min_len, max_len = 10, 100
    num_samples = len(cleaned_task_data)

    def dfs_path(start, end, detour_path, origin_path):
        stack = [(start, [start])]
        paths = []
        while stack:
            (vertex, path) = stack.pop()

            if vertex == end and path != detour_path:  # 如果到达终点，则将路径添加到结果列表中
                paths.append(path)
                # after_detour_length = np.sum([road_length_dict[road] for road in path])
                # before_detour_length = np.sum([road_length_dict[road] for road in detour_path])
                # path_length = np.sum([road_length_dict[road] for road in origin_path])

                poly = detour_path[::-1][:-1] + path
                pt_list = []
                for road in poly:
                    pt_list += [float(item) for pair in geometry_df.iloc[road]['geometry'][12:-1].split(', ') for item
                                in pair.split(' ')]
                area = Polygon([[pt_list[i], pt_list[i + 1]] for i in range(0, len(pt_list), 2)]).area

                if area > 1e-6:
                    global count
                    count += 1
                    #  print(origin_path)
                    #  print(detour_path)
                    #  print(path)
                    #  print(path_length, after_detour_length, before_detour_length , origin_path_length)
                    return path

            if len(path) - 1 == int(1 / 3 * len(origin_path)) + 1:  # 如果长度到达最大长度但不符合标准则去除
                continue

            for neighbor in torch.nonzero(trans_mat[vertex] != 0).reshape(-1, ).numpy().tolist():
                if neighbor not in path:  # 如果邻居节点尚未访问过，则将其添加到路径中，并将其压入栈中
                    stack.append((neighbor, path + [neighbor]))
        if len(paths) == 0:
            return None
        return paths[-1]

    def detour(replace_rate, path, tm_list, detour_path, start_pos):
        # 需要重建detour_base
        # 开始和结束的位置不变
        start_time = tm_list[0]
        detour_anchor = dfs_path(detour_path[0], detour_path[-1], detour_path, path)
        if detour_anchor is None:
            detour_anchor = detour_path
        end_pos = start_pos + len(detour_path)
        pre_path = path[:start_pos]
        next_path = path[end_pos:]
        p = np.random.random_sample()  # 产生[0,1)之间的随机数
        if p > replace_rate:
            new_path = pre_path + detour_anchor + next_path
        else:
            new_path = pre_path + [padding_id] * len(detour_path) + next_path

        # 产生tm_list
        cur_travel_time_dict = {}
        for i in range(len(path) - 1):
            cur_travel_time_dict[path[i]] = tm_list[i + 1] - tm_list[i]

        tm_list = [start_time]
        for road in new_path:
            shift_ratio = np.random.uniform(-0.1, 0.1)
            if road in cur_travel_time_dict.keys():
                tm_list.append(tm_list[-1] + cur_travel_time_dict[road] * (1 + shift_ratio))
            else:
                tm_list.append(tm_list[-1] + history_travel_time_dict[road] * (1 + shift_ratio))
        return new_path, tm_list

    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)
    week_arr = np.full([num_queries, max_len], 0, dtype=np.int32)
    minute_arr = np.full([num_queries, max_len], 0, dtype=np.int32)

    query_route_length = []
    for i in tqdm(range(num_queries)):
        row = cleaned_task_data.iloc[random_index[i]]
        sample_len = int(row['route_length'] * detour_rate) + 1
        path = row['cpath_list']
        try_count = 0
        sample_pos_list = list(range(1, row['route_length'] - sample_len, 1))
        while path == row['cpath_list'] and try_count < 10 and len(sample_pos_list) != 0:
            try_count += 1
            start_pos = np.random.choice(sample_pos_list, 1)[0]  # OD不参与处理
            sample_pos_list.remove(start_pos)
            detour_path = row['cpath_list'][start_pos:start_pos + sample_len]
            # detour_travel_time = row['road_timestamp'][start_pos+sample_len-1] - row['road_timestamp'][start_pos]
            origin_path_length = row['total_length']
            path, tm_list = detour(0, row['cpath_list'], row['road_timestamp'], detour_path, start_pos)

        week_list = []
        minute_list = []
        for tm in tm_list[0:-1]:
            dt = datetime.fromtimestamp(tm)
            weekday = dt.weekday() + 1
            minute = dt.hour * 60 + dt.minute + 1
            week_list.append(weekday)
            minute_list.append(minute)

        query_route_length.append(len(path))
        q_arr[i, :len(path)] = np.array(path, dtype=np.int32)
        week_arr[i, :len(path)] = np.array(week_list, dtype=np.int32)
        minute_arr[i, :len(path)] = np.array(minute_list, dtype=np.int32)

    y = random_index[:num_queries]
    tm_arr = torch.cat([torch.LongTensor(week_arr).unsqueeze(dim=2), torch.LongTensor(minute_arr).unsqueeze(dim=2),
                        torch.zeros_like(torch.LongTensor(minute_arr).unsqueeze(dim=2)).long()], dim=-1)

    return torch.LongTensor(q_arr), y, np.array(query_route_length), tm_arr

def evaluation(seq_embedding, road_embedding, seq_model, task_data, num_nodes, detour_base, detour_rate, fold=5):
    print('device: cpu')
    batch_size = 1024
    num_queries = 5000

    cleaned_split_df = task_data

    x = seq_embedding.cpu()

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)

    hit_list, mean_rank_list, no_hit_list = [], [], []
    for expid in range(fold):
        queries, y, query_route_length = query_prepare3(expid, cleaned_split_df, num_nodes, num_queries, detour_base, detour_rate)

        # q = []
        # for batch_idx in next_batch_index(num_queries, batch_size, shuffle=False):
        #     q_batch = queries[batch_idx] # route_assign_mat
        #     length_batch = query_route_length[batch_idx]
        #     # _, traj_rep = seq_model.encode_route(None, q_batch, q_batch)
        #     seq_rep = get_seq_emb_from_node(road_embedding, q_batch, length_batch, batch_size)
        #     q.append(seq_rep)
        # q = torch.cat(q, dim=0).detach().cpu().numpy()

        # 数量少可以直接推理
        if road_embedding is None:
            route_assign_mat = queries.long().cuda()
            route_data = torch.zeros((route_assign_mat.shape[0], route_assign_mat.shape[1], 2)).long().cuda()
            queries = (route_data, route_assign_mat, None, None, route_assign_mat, None, None)
            q = get_seq_emb_from_traj_withRouteOnly(seq_model, queries, batch_size=1024)
        else:
            q = get_seq_emb_from_node(road_embedding, queries, query_route_length, batch_size)

        q = q.cpu().numpy()

        D, I = index.search(q, 1000)  # D是距离,I是index的id

        hit = 0
        rank_sum = 0
        no_hit = 0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:10]:
                    hit += 1
            else:
                no_hit += 1

        hit_list.append(hit / (num_queries - no_hit))

        mean_rank_list.append(rank_sum / num_queries)
        no_hit_list.append(no_hit)

        print(f'exp {expid} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}')

    print(f'similarity search       | Mean Rank: {np.mean(mean_rank_list):.4f}, HR@10: {np.mean(hit_list):.4f}, No Hit: {np.mean(no_hit_list)}')

# 带时间的
def evaluation2(seq_embedding, road_embedding, seq_model, task_data, num_nodes, detour_base, feature_df, detour_rate, fold=5):
    print('device: cpu')
    batch_size = 1024
    num_queries = 5000

    cleaned_split_df = task_data

    seq_embedding = F.normalize(seq_embedding, dim=1)
    x = seq_embedding.cpu()


    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)

    hit_list, mean_rank_list, no_hit_list = [], [], []
    for expid in range(fold):
        queries, y, query_route_length, tm_mat = query_prepare4(expid, cleaned_split_df, num_nodes, num_queries, detour_base, feature_df, detour_rate)

        # q = []
        # for batch_idx in next_batch_index(num_queries, batch_size, shuffle=False):
        #     q_batch = queries[batch_idx] # route_assign_mat
        #     length_batch = query_route_length[batch_idx]
        #     # _, traj_rep = seq_model.encode_route(None, q_batch, q_batch)
        #     seq_rep = get_seq_emb_from_node(road_embedding, q_batch, length_batch, batch_size)
        #     q.append(seq_rep)
        # q = torch.cat(q, dim=0).detach().cpu().numpy()

        # 数量少可以直接推理
        if road_embedding is None:
            route_assign_mat = queries.long().cuda()
            route_data = tm_mat.long().cuda()
            queries = (route_data, route_assign_mat, None, None, route_assign_mat, None, None)
            q = get_seq_emb_from_traj_withRouteOnly(seq_model, queries, batch_size=1024)
        else:
            q = get_seq_emb_from_node(road_embedding, queries, query_route_length, batch_size)

        q = F.normalize(q, dim=1)
        q = q.cpu().numpy()

        D, I = index.search(q, 1000)  # D是距离,I是index的id

        hit = 0
        rank_sum = 0
        no_hit = 0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:10]:
                    hit += 1
            else:
                no_hit += 1

        hit_list.append(hit / (num_queries - no_hit))

        mean_rank_list.append(rank_sum / num_queries)
        no_hit_list.append(no_hit)

        print(f'exp {expid} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}')

    print(f'similarity search       | Mean Rank: {np.mean(mean_rank_list):.4f}, HR@10: {np.mean(hit_list):.4f}, No Hit: {np.mean(no_hit_list)}')

# 带时间的 连续子序列detour
def evaluation3(seq_embedding, road_embedding, seq_model, task_data, num_nodes, detour_base, feature_df, geometry_df, detour_rate, fold=5):

    print('device: cpu')
    batch_size = 1024
    num_queries = 5000

    cleaned_split_df = task_data

    seq_embedding = F.normalize(seq_embedding, dim=1)
    x = seq_embedding.cpu()

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)

    hit_list, mean_rank_list, no_hit_list = [], [], []
    for expid in range(fold):
        queries, y, query_route_length, tm_mat = query_prepare5(expid, cleaned_split_df, num_nodes, num_queries, detour_base, feature_df,geometry_df, detour_rate)

        # q = []
        # for batch_idx in next_batch_index(num_queries, batch_size, shuffle=False):
        #     q_batch = queries[batch_idx] # route_assign_mat
        #     length_batch = query_route_length[batch_idx]
        #     # _, traj_rep = seq_model.encode_route(None, q_batch, q_batch)
        #     seq_rep = get_seq_emb_from_node(road_embedding, q_batch, length_batch, batch_size)
        #     q.append(seq_rep)
        # q = torch.cat(q, dim=0).detach().cpu().numpy()

        # 数量少可以直接推理
        if road_embedding is None:
            route_assign_mat = queries.long().cuda()
            route_data = tm_mat.long().cuda()
            queries = (route_data, route_assign_mat, None, None, route_assign_mat, None, None)
            q = get_seq_emb_from_traj_withRouteOnly(seq_model, queries, batch_size=1024)
        else:
            q = get_seq_emb_from_node(road_embedding, queries, query_route_length, batch_size=1024)

        q = F.normalize(q, dim=1)
        q = q.cpu().numpy()

        D, I = index.search(q, 1000)  # D是距离,I是index的id

        hit = 0
        rank_sum = 0
        no_hit = 0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:10]:
                    hit += 1
            else:
                no_hit += 1

        hit_list.append(hit / (num_queries - no_hit))

        mean_rank_list.append(rank_sum / num_queries)
        no_hit_list.append(no_hit)

        print(f'exp {expid} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}')

    print(f'similarity search       | Mean Rank: {np.mean(mean_rank_list):.4f}, HR@10: {np.mean(hit_list):.4f}, No Hit: {np.mean(no_hit_list)}')


class Projector(nn.Module):
    def __init__(self, input_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.activation = nn.ReLU()
        self.cls = nn.Linear(input_size*2, 1)

    def forward(self, x1, x2):
        rep1 = self.activation(self.fc(x1))
        rep2 = self.activation(self.fc(x2))
        rep = torch.cat([rep1, rep2], dim=1)
        pred = self.cls(rep)
        return pred

def finetune_evaluation2(seq_embedding, road_embedding, seq_model, task_data, num_nodes, detour_base, feature_df, detour_rate, fold=5):
    print('device: cpu')
    batch_size = 1024
    num_queries = 5000

    cleaned_split_df = task_data

    num_finetunes = 500
    projector = Projector(seq_embedding.shape[-1]).cuda()
    queries, y, query_route_length, tm_mat = query_prepare4(9999, cleaned_split_df, num_nodes, num_finetunes,
                                                            detour_base, feature_df, detour_rate)
    route_assign_mat = queries.long().cuda()
    route_data = tm_mat.long().cuda()
    queries = (route_data, route_assign_mat, None, None, route_assign_mat, None, None)
    detour_rep = get_seq_emb_from_traj_withRouteOnly(seq_model, queries, batch_size=1024).cuda()
    origin_rep = seq_embedding[y].cuda()

    # 对于每个样本构造5个负样本
    label = []
    x1 = []
    x2 = []
    for i in range(num_finetunes):
        label.append(1)
        x1.append(origin_rep[i])
        x2.append(detour_rep[i])
        # 构造负样本
        sample_pos = [item for item in list(range(num_finetunes)) if item != i]
        neg_pos = np.random.sample(sample_pos, 3)
        for pos in neg_pos:
            x1.append(origin_rep[i])
            x2.append(detour_rep[pos])
            label.append(0)
    label = torch.tensor(label).float().cuda()
    x1 = torch.stack(x1, dim=0).cuda()
    x2 = torch.stack(x2, dim=0).cuda()
    opt = torch.optim.Adam(projector.parameters(), lr=1e-2)

    for _ in range(100):
        projector.train()
        pred = projector(x1, x2)
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(pred.squeeze(-1), label)
        print(loss)
        loss.backward()
        opt.step()

    # x = projector(seq_embedding).detach().cpu().numpy()

    x = projector.activation(projector.fc(x1)).detach().cpu().numpy()

    index = faiss.IndexFlatL2(x.shape[1]) # faiss.IndexFlatIP是内积 ；faiss.indexFlatL2是欧式距离
    index.add(x)

    hit_list, mean_rank_list, no_hit_list = [], [], []
    for expid in range(fold):
        queries, y, query_route_length, tm_mat = query_prepare4(expid, cleaned_split_df, num_nodes, num_queries, detour_base, feature_df, detour_rate)

        # q = []
        # for batch_idx in next_batch_index(num_queries, batch_size, shuffle=False):
        #     q_batch = queries[batch_idx] # route_assign_mat
        #     length_batch = query_route_length[batch_idx]
        #     # _, traj_rep = seq_model.encode_route(None, q_batch, q_batch)
        #     seq_rep = get_seq_emb_from_node(road_embedding, q_batch, length_batch, batch_size)
        #     q.append(seq_rep)
        # q = torch.cat(q, dim=0).detach().cpu().numpy()

        # 数量少可以直接推理
        if road_embedding is None:
            route_assign_mat = queries.long().cuda()
            route_data = tm_mat.long().cuda()
            queries = (route_data, route_assign_mat, None, None, route_assign_mat, None, None)
            q = get_seq_emb_from_traj_withRouteOnly(seq_model, queries, batch_size=1024)
        else:
            q = get_seq_emb_from_node(road_embedding, queries, query_route_length, batch_size)

        # q = projector(q).detach()
        q = projector.activation(projector.fc(q)).detach()
        q = q.cpu().numpy()

        D, I = index.search(q, 1000)  # D是距离,I是index的id

        hit = 0
        rank_sum = 0
        no_hit = 0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:10]:
                    hit += 1
            else:
                no_hit += 1

        hit_list.append(hit / (num_queries - no_hit))

        mean_rank_list.append(rank_sum / num_queries)
        no_hit_list.append(no_hit)

        print(f'exp {expid} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}')

    print(f'similarity search       | Mean Rank: {np.mean(mean_rank_list):.4f}, HR@10: {np.mean(hit_list):.4f}, No Hit: {np.mean(no_hit_list)}')
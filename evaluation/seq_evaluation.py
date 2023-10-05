import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import argparse
from JTMR import JMTRModel
from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, prepare_data
import torch
import os
torch.set_num_threads(5)

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def evaluation(city, exp_path, model_name, start_time):
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]

    # load task 1 & task2 label
    feature_df = pd.read_csv("/data/mazp/dataset/JMTR/didi_{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("/data/mazp/dataset/JMTR/didi_{}/line_graph_edge_idx.npy".format(city))
    print("edge_index shape:", edge_index.shape)

    #
    test_node_data = pickle.load(
        open('/data/mazp/dataset/JMTR/didi_{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb')) # data_seq_evaluation.pkl

    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    # prepare
    num_samples = 'all'
    if num_samples == 'all':
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)

    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] # model.to()包含inplace操作，不需要对象承接
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    test_seq_data = pickle.load(
        open('/data/mazp/dataset/JMTR/didi_{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city),
             'rb'))
    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_length = test_seq_data['route_length'].values
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)

    # task 3
    time_est.evaluation(seq_embedding, test_seq_data, num_nodes)

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_data = (
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)


    # task 4
    # detour_base = pickle.load(
    #     open('/data/mazp/dataset/JMTR/didi_{}/detour_base_max5.pkl'.format(city), 'rb'))
    #
    # sim_srh.evaluation2(seq_embedding, None, seq_model, test_seq_data, num_nodes, detour_base, feature_df,
    #                     detour_rate=0.15, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    geometry_df = pd.read_csv("/data/mazp/dataset/JMTR/didi_{}/edge_geometry.csv".format(city))

    trans_mat = np.load('/data/mazp/dataset/JMTR/didi_{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(seq_embedding, None, seq_model, test_seq_data, num_nodes, trans_mat, feature_df, geometry_df,
                        detour_rate=0.3, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    city = 'chengdu'

    # xian
    # exp_path = '/data/mazp/exp/JTMR_xian_230811074048'
    # model_name = 'JTMR_xian_v1_20_100000_230811074048_19.pt'

    # chengdu
    # exp_path = '/data/mazp/exp/JTMR_chengdu_230812005640'
    # model_name = 'JTMR_chengdu_v1_20_100000_230812005640_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_chengdu_230817040154'
    # model_name = 'JTMR_chengdu_v1_20_100000_230817040154_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_xian_230818192648'
    # model_name = 'JTMR_xian_v1_20_100000_230818192648_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_xian_230819055718'
    # model_name = 'JTMR_xian_v1_20_100000_230819055718_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_chengdu_230819202102'
    # model_name = 'JTMR_chengdu_v1_20_100000_230819202102_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_chengdu_230822074358'
    # model_name = 'JTMR_chengdu_v1_20_100000_230822074358_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_xian_230822140316'
    # model_name = 'JTMR_xian_v1_20_100000_230822140316_19.pt'
    #
    # exp_path = '/data/mazp/exp/JTMR_chengdu_230822153727'
    # model_name = 'JTMR_chengdu_v1_20_100000_230822153727_19.pt'

    # exp_path = '/data/mazp/exp/JTMR_xian_230823032506'
    # model_name = 'JTMR_xian_v1_20_100000_230823032506_19.pt'

    # exp_path = '/data/mazp/exp/JTMR_xian_230824022601'
    # model_name = 'JTMR_xian_v1_20_100000_230824022601_19.pt'

    # exp_path = '/data/mazp/exp/JTMR_xian_230824094211'
    # model_name = 'JTMR_xian_v1_20_100000_230824094211_19.pt'



    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(city, exp_path, model_name, start_time)

# xian
# travel time estimation  | MAE: 87.3313, RMSE: 119.3156 # 全置为0
# travel time estimation  | MAE: 87.6380, RMSE: 119.6562 # min 置为0 相似度低
# travel time estimation  | MAE: 87.0120, RMSE: 118.7117 # week置为0 相似度高

# chengdu
# travel time estimation  | MAE: 83.7680, RMSE: 111.8366 # 全置为0
# travel time estimation  | MAE: 84.4726, RMSE: 111.8322 # min 置为0 相似度低
# travel time estimation  | MAE: 84.0630, RMSE: 111.7760 # week 置为0 相似度低

# travel time estimation  | MAE: 83.8837, RMSE: 111.5713
# similarity search       | Mean Rank: 6.9435, HR@10: 0.8548, No Hit: 0.0

# travel time estimation  | MAE: 87.1631, RMSE: 119.0558
# similarity search       | Mean Rank: 9.3613, HR@10: 0.8039, No Hit: 0.0
# similarity search       | Mean Rank: 8.6928, HR@10: 0.8197, No Hit: 0.0 # min和delta为0
# similarity search       | Mean Rank: 9.2603, HR@10: 0.8053, No Hit: 0.0 # 全0

# xian
# exp_path = '/data/mazp/exp/JTMR_xian_230819055718'
# model_name = 'JTMR_xian_v1_20_100000_230819055718_19.pt'
# travel time estimation  | MAE: 87.1591, RMSE: 118.8026 # 间隔和分钟置为mask值
# similarity search       | Mean Rank: 6.0727, HR@10: 0.8828, No Hit: 0.0 # 不做特殊处理


# chengdu
# travel time estimation  | MAE: 83.8969, RMSE: 111.8088
# similarity search       | Mean Rank: 6.4462, HR@10: 0.8665, No Hit: 0.0

# chengdu 对比 调整tau
# travel time estimation  | MAE: 83.7518, RMSE: 111.8767
# similarity search       | Mean Rank: 6.4036, HR@10: 0.8652, No Hit: 0.0

# xian 对比 调整tau
# travel time estimation  | MAE: 87.2686, RMSE: 119.1458
# similarity search       | Mean Rank: 6.0667, HR@10: 0.8811, No Hit: 0.0

# chengdu 对比 三角损失
# travel time estimation  | MAE: 83.7620, RMSE: 111.7484
# similarity search       | Mean Rank: 6.5985, HR@10: 0.8657, No Hit: 0.0
# 加norm后结果好一些
# travel time estimation  | MAE: 84.3132, RMSE: 112.0723
# similarity search       | Mean Rank: 5.9975, HR@10: 0.8704, No Hit: 0.0  ind pair detour detour_rate = 0.15
# similarity search       | Mean Rank: 5.8675, HR@10: 0.8870, No Hit: 0.0  cont sub traj detour detour_rate = 0.25
# similarity search       | Mean Rank: 2.7228, HR@10: 0.9424, No Hit: 0.0  cont sub traj detour detour_rate = 0.1
# similarity search       | Mean Rank: 4.3831, HR@10: 0.9182, No Hit: 0.0  cont sub traj detour detour_rate = 0.2
# similarity search       | Mean Rank: 6.6062, HR@10: 0.8689, No Hit: 0.0  cont sub traj detour detour_rate = 0.3
# similarity search       | Mean Rank: 9.6757, HR@10: 0.8199, No Hit: 0.0  cont sub traj detour detour_rate = 0.4

# new detour
# similarity search       | Mean Rank: 3.7809, HR@10: 0.9140, No Hit: 0.0 # 0.2
# similarity search       | Mean Rank: 2.5484, HR@10: 0.9411, No Hit: 0.0 # 0.3

# xian 对比 三角损失
# travel time estimation  | MAE: 87.1660, RMSE: 119.2541
# similarity search       | Mean Rank: 5.9897, HR@10: 0.8864, No Hit: 0.3
# 加norm后结果好一些
# travel time estimation  | MAE: 87.2475, RMSE: 119.3162
# similarity search       | Mean Rank: 5.6661, HR@10: 0.8904, No Hit: 0.0 ind pair detour detour_rate = 0.15
# similarity search       | Mean Rank: 3.8581, HR@10: 0.9232, No Hit: 0.0 cont sub traj detour detour_rate = 0.2
# similarity search       | Mean Rank: 5.8871, HR@10: 0.8924, No Hit: 0.0 cont sub traj detour detour_rate = 0.25

# new detour
# similarity search       | Mean Rank: 4.2792, HR@10: 0.8964, No Hit: 0.0 0.2
# similarity search       | Mean Rank: 2.7714, HR@10: 0.9294, No Hit: 0.0 0.3








# xian 对比 三角损失 + 1*轨迹级对比 CL的加入会有一些负面作用
# travel time estimation  | MAE: 95.5387, RMSE: 127.6472
# similarity search       | Mean Rank: 54.1119, HR@10: 0.7372, No Hit: 1244.3

# xian 对比 三角损失 + 0.1*轨迹级对比 CL的加入会有一些负面作用
# travel time estimation  | MAE: 89.2120, RMSE: 121.4869
# similarity search       | Mean Rank: 13.1270, HR@10: 0.8667, No Hit: 19.8




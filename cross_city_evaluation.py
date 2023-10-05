# todo
# 分别使用成都和西安训练好的模型，使用随机初始化的embedding获得路段表示，然后在下游任务上评估

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import time
import pickle
from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_road_emb_from_traj, prepare_data, get_seq_emb_from_node, get_seq_emb_from_traj_withRouteOnly
import torch
import os
torch.set_num_threads(5)

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

graph_setting = {
    'chengdu':6450,
    'xian':4996
}

def evaluation(source_city, target_city, exp_path, model_name, start_time):
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]

    # load task 1 & task2 label
    feature_df = pd.read_csv("D:/research/dataset/JMTR/didi_{}/edge_features.csv".format(target_city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("D:/research/dataset/JMTR/didi_{}/line_graph_edge_idx.npy".format(target_city))
    print("edge_index shape:", edge_index.shape)

    # load origin train data
    test_node_data = pickle.load(
        open('D:/research/dataset/JMTR/didi_{}/{}_1101_1115_data_sample10w.pkl'.format(target_city, target_city), 'rb'))
    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    # sample train data
    num_samples = 'all' # 'all' or 50000
    if num_samples == 'all':
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)
    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    # load model
    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']
    _, route_embed_size = seq_model.node_embedding.weight.shape
    seq_model.vocab_size = graph_setting[target_city]
    seq_model.node_embedding = torch.nn.Embedding(seq_model.vocab_size, route_embed_size)
    seq_model.node_embedding.requires_grad_(False)
    seq_model.node_embedding.cuda()
    seq_model.edge_index = torch.tensor(edge_index).cuda()

    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare road task dataset
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_node_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_node_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)

    update_road = 'route'
    emb_path = 'D:/research/dataset/JMTR/didi_{}/{}_1101_1115_road_embedding_{}_{}_{}.pkl'.format(
        target_city, target_city, embedding_name, num_samples, update_road)

    if os.path.exists(emb_path):
        # load road embedding from inference result
        road_embedding = torch.load(emb_path, map_location='cuda:{}'.format(dev_id))['road_embedding']
    else:
        # infer road embedding
        road_embedding = get_road_emb_from_traj(seq_model, test_node_data, without_gps=False, batch_size=256,
                                                update_road=update_road, city=target_city)
        torch.save({'road_embedding': road_embedding}, emb_path)

    # task 1
    road_cls.evaluation(road_embedding, feature_df)

    # task 2
    speed_inf.evaluation(road_embedding, feature_df)

    # prepare sequence task
    test_seq_data = pickle.load(
        open('D:/research/dataset/JMTR/didi_{}/{}_1101_1115_data_seq_evaluation.pkl'.format(target_city, target_city),
             'rb'))
    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_length = test_seq_data['route_length'].values
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    test_data = (
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
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

    geometry_df = pd.read_csv("D:/research/dataset/JMTR/didi_{}/edge_geometry.csv".format(target_city))

    trans_mat = np.load('D:/research/dataset/JMTR/didi_{}/transition_prob_mat.npy'.format(target_city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(seq_embedding, None, seq_model, test_seq_data, num_nodes, trans_mat, feature_df, geometry_df,
                        detour_rate=0.3, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空


    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    # source_city = 'chengdu'
    # target_city = 'xian'
    #
    # exp_path = 'D:/research/exp/JTMR_chengdu_230822153727'
    # model_name = 'JTMR_chengdu_v1_20_100000_230822153727_19.pt'

    source_city = 'xian'
    target_city = 'chengdu'

    exp_path = 'D:/research/exp/JTMR_xian_230823032506'
    model_name = 'JTMR_xian_v1_20_100000_230823032506_19.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(source_city, target_city, exp_path, model_name, start_time)

# chengdu --> xian zero-shot
# road classification     | micro F1: 0.7252, macro F1: 0.6873
# travel speed estimation | MAE: 2.2425, RMSE: 3.1753
# travel time estimation  | MAE: 109.2060, RMSE: 141.6533
# similarity search       | Mean Rank: 79.6869, HR@10: 0.4534, No Hit: 724.5

# chengdu --> xian finetune 5k trajectory
# road classification     | micro F1: 0.7295, macro F1: 0.6916
# travel speed estimation | MAE: 2.4765, RMSE: 3.4221
# travel time estimation  | MAE: 106.5079, RMSE: 139.2584
# similarity search       | Mean Rank: 64.6180, HR@10: 0.5007, No Hit: 249.3

# xian --> chengdu zero-shot
# road classification     | micro F1: 0.6712, macro F1: 0.6662
# travel speed estimation | MAE: 2.6491, RMSE: 3.6021
# travel time estimation  | MAE: 105.2994, RMSE: 134.9308
# similarity search       | Mean Rank: 61.8339, HR@10: 0.5277, No Hit: 534.6

# xian --> chengdu finetune 5k trajectory
# road classification     | micro F1: 0.6802, macro F1: 0.6779
# travel speed estimation | MAE: 2.8148, RMSE: 3.7797
# travel time estimation  | MAE: 99.1057, RMSE: 128.7578
# similarity search       | Mean Rank: 53.6787, HR@10: 0.5632, No Hit: 190.8


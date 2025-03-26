import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from utils import weight_init
from dataloader import get_train_loader, random_mask
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from JTTT_withmatch import JTTTModel
from cl_loss import get_traj_cl_loss, get_road_cl_loss,get_traj_cluster_loss,get_traj_match_loss
from dcl import DCL
import os
dev_id = 0
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)
torch.set_num_threads(10)

def finetune(config):

    source_city = config['source_city']
    target_city = config['target_city']
    exp_path = config['exp_path']
    model_name = config['model_name']

    vocab_size = config['vocab_size']
    num_samples = config['num_samples']
    data_path = config['data_path']
    adj_path = config['adj_path']

    save_path = config['save_path']

    num_worker = config['num_worker']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    route_min_len = config['route_min_len']
    route_max_len = config['route_max_len']
    gps_min_len = config['gps_min_len']
    gps_max_len = config['gps_max_len']

    hidden_size = config['hidden_size']

    verbose = config['verbose']
    version = config['version']
    seed = config['random_seed']

    mask_length = config['mask_length']
    mask_prob = config['mask_prob']

    # 设置随机种子
    setup_seed(seed)

    edge_index = np.load(adj_path)
    model_path = os.path.join(exp_path, 'model', model_name)

    # load model
    model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']
    _, route_embed_size = model.node_embedding.weight.shape
    model.vocab_size = vocab_size
    model.node_embedding = torch.nn.Embedding(vocab_size, route_embed_size)
    model.node_embedding.requires_grad_(True)
    model.node_embedding.cuda()
    model.edge_index = torch.tensor(edge_index).cuda()
    model.gps_mlm_head = nn.Linear(hidden_size, vocab_size)
    model.route_mlm_head = nn.Linear(hidden_size, vocab_size)
    model.gps_mlm_head.cuda()
    model.route_mlm_head.cuda()

    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'JTMR_{}_{}_{}_{}_{}'.format(target_city, version, num_epochs, num_samples, nowtime)
    model_path = os.path.join(save_path, 'JTMR_{}_{}'.format(target_city, nowtime), 'model')
    log_path = os.path.join(save_path, 'JTMR_{}_{}'.format(target_city, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    train_loader = get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed)
    print('dataset is ready.')

    epoch_step = train_loader.dataset.route_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = batch

            masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                         vocab_size, mask_length, mask_prob)

            route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length =\
                route_data.cuda(), masked_route_assign_mat.cuda(), gps_data.cuda(), masked_gps_assign_mat.cuda(), route_assign_mat.cuda(), gps_length.cuda()

            gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep \
                = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length)

            # flatten road_rep
            mat2flatten = {}
            y_label = []
            route_length = (route_assign_mat != model.vocab_size).int().sum(1)
            gps_road_list, route_road_list, gps_road_joint_list, route_road_joint_list = [], [], [], []
            now_flatten_idx = 0
            for i, length in enumerate(route_length):
                y_label.append(route_assign_mat[i, :length]) # route 和 gps mask 位置是一样的
                gps_road_list.append(gps_road_rep[i, :length])
                route_road_list.append(route_road_rep[i, :length])
                gps_road_joint_list.append(gps_road_joint_rep[i, :length])
                route_road_joint_list.append(route_road_joint_rep[i, :length])
                for l in range(length):
                    mat2flatten[(i, l)] = now_flatten_idx
                    now_flatten_idx += 1

            y_label = torch.cat(y_label, dim=0)
            gps_road_rep = torch.cat(gps_road_list, dim=0) # road_num
            route_road_rep = torch.cat(route_road_list, dim=0)
            gps_road_joint_rep = torch.cat(gps_road_joint_list, dim=0)
            route_road_joint_rep = torch.cat(route_road_joint_list, dim=0)

            # project rep into the same space
            gps_traj_rep = model.gps_proj_head(gps_traj_rep)
            route_traj_rep = model.route_proj_head(route_traj_rep)

            # (GRM) get gps & route rep matching loss
            tau = 0.07
            match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)

            # MLM task supervised
            # prepare label and mask_pos
            masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
            masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
            y_label = y_label[masked_pos].long()

            # 事实上SpeechLM中的swap是在shared transformer之前进行的，为的是将token对应词的embedding和固定长度因素得到的音素单元部分对齐的方法，类似于添加锚点

            # (MLM 1) get gps rep road loss
            gps_mlm_pred = model.gps_mlm_head(gps_road_joint_rep) # project head 也会被更新
            masked_gps_mlm_pred = gps_mlm_pred[masked_pos]
            gps_mlm_loss = nn.CrossEntropyLoss()(masked_gps_mlm_pred, y_label)

            # (MLM 2) get route rep road loss
            route_mlm_pred = model.route_mlm_head(route_road_joint_rep) # project head 也会被更新
            masked_route_mlm_pred = route_mlm_pred[masked_pos]
            route_mlm_loss = nn.CrossEntropyLoss()(masked_route_mlm_pred, y_label)

            # loss = (route_mlm_loss + gps_mlm_loss + 0.2*traj_cl_loss + 0.1*road_cl_loss)/4
            # loss = (route_mlm_loss + gps_mlm_loss + 0.2*traj_cl_loss) / 3
            loss = (route_mlm_loss + gps_mlm_loss + 2*match_loss) / 3

            step = epoch_step*epoch + idx
            writer.add_scalar('match_loss/match_loss', match_loss, step)
            writer.add_scalar('mlm_loss/gps_mlm_loss', gps_mlm_loss, step)
            writer.add_scalar('mlm_loss/route_mlm_loss', route_mlm_loss, step)
            writer.add_scalar('loss', loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (idx + 1) % verbose:
                t = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={loss.item():.4f}')

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))

    return model

if __name__ == '__main__':
    source_city = 'chengdu'
    target_city = 'xian'
    config = json.load(open('config/finetune_{}2{}.json'.format(source_city, target_city), 'r'))
    finetune(config)
    

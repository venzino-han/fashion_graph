import copy, math
import pickle as pkl
from easydict import EasyDict
from prettytable import PrettyTable
import time

import dgl
import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

import config
from models.frgcn import FRGCN
from models.fgat import FGAT
from models.flgcn import FLGCN
from data_generator_task1 import get_task1_dataloader
from utils import get_args_from_yaml, get_logger


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# TODO Task 2 evaluation method --> top 100
def get_rank(iids, answer_iid):
    if answer_iid not in set(iids):
        return 101
    else:
        rank = iids.index(answer_iid) +1
        return rank 

task2_valid_query_df = pd.read_csv('./processed_data/itemset_item_valid_query.csv')
task2_valid_answer_df = task2_valid_query_df.query('answer==1')

def evaluate_task2(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_preds = []
    for batch in loader:
        with th.no_grad():
            preds, _ = model(batch[0].to(device))
        val_preds.extend(preds.cpu().tolist())

    itemset_item_answer_dict={k:v for k,v in zip(task2_valid_answer_df.itemset_id, task2_valid_answer_df.item_id)}

    preds_df = pd.DataFrame({
        'itemset_id': task2_valid_query_df.itemset_id,
        'item_id': task2_valid_query_df.item_id,
        'score': val_preds,
    })

    accs = []
    hit = []
    for itemset_id, sub_df in preds_df.groupby('itemset_id'):
        sub_df = sub_df.sort_values('score', ascending=False)
        pred_iid = list(sub_df.item_id)[0]
        answer_iid = itemset_item_answer_dict[itemset_id]
        if pred_iid != answer_iid:
            accs.append(0)
            hit.append(0)
        else:
            accs.append(1)
            hit.append(1)
    val_result = np.mean(hit)
    val_acc = np.mean(accs)
    print('val hit@3: ', val_result)
    return None, val_result, val_acc

def evaluate(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())

    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, None, val_acc


def train_epoch(model, optimizer, loader, device, logger, log_interval):
    model.train()

    epoch_loss = 0.0
    iter_loss = 0.0
    iter_mse = 0.0
    iter_cnt = 0
    iter_dur = []
    mse_loss_fn = nn.MSELoss().to(device)

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(inputs)
        optimizer.zero_grad()
        mse_loss = mse_loss_fn(preds, labels)
        loss = mse_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(
                f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f},"
            )
            iter_loss = 0.0
            iter_mse = 0.0
            iter_cnt = 0
            if iter_idx == 500:
                break

    return epoch_loss / len(loader.dataset)

MODEL_MAP = {
    "FRGCN": FRGCN,
    "FGAT": FGAT,
    "FLGCN": FLGCN,
}

def train(args: EasyDict, train_loader, test_loader, logger):
    th.manual_seed(0)
    np.random.seed(0)

    in_feats = args.model_input_feat_dims
    model_class = MODEL_MAP.get(args.model_type)
    model = model_class(
        input_dims=in_feats,
        hidden_dims=args.model_hidden_dims,
        num_layers=args.model_num_layers,
        trans_pooling=args.model_trans_pooling,
    ).to(args.train_device)

    if args.get("model_parameters") is not None:
        model.load_state_dict(th.load(f"./parameters/{args.model_parameters}"))

    optimizer = optim.Adam(
        model.parameters(), lr=args.train_learning_rates, weight_decay=args.train_weight_decay
    )
    logger.info("Loading network finished ...\n")
    count_parameters(model)

    best_epoch = 0
    best_auc, best_acc = 0, -1

    logger.info(f"Start training ... learning rate : {args.train_learning_rates}")
    epochs = list(range(1, args.train_epochs + 1))

    eval_func_map = {
        "task1": evaluate,
        "task2": evaluate_task2,
    }
    eval_func = eval_func_map.get(args.dataset_name, evaluate)
    for epoch_idx in epochs:
        logger.debug(f"Epoch : {epoch_idx}")

        train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            args.train_device,
            logger,
            log_interval=args.log_interval,
        )
        # train_loss = 0
        test_auc, test_rank, test_acc = eval_func(model, test_loader, args.train_device)
        if test_auc is None:
            test_auc = -1
        if test_rank is None:
            test_rank = -1

        logger.info(
            f"=== Epoch {epoch_idx}, train loss {train_loss:.4f}, test auc {test_auc:.4f}, test acc {test_acc:.4f} ==="
        )

        if epoch_idx % args.train_lr_decay_step == 0:
            for param in optimizer.param_groups:
                param["lr"] = args.train_lr_decay_factor * param["lr"]
            print("lr : ", param["lr"])

        if test_rank == -1 :
            if best_auc < test_auc:
                logger.info(f'new best test auc {test_auc:.4f} acc {test_acc:.4f} ===')
                best_epoch = epoch_idx
                best_auc = test_auc
                best_acc = test_acc
                best_lr = args.train_learning_rates
                best_state = copy.deepcopy(model.state_dict())
        else:
            if best_acc < test_acc:
                logger.info(f'new best test rank {test_rank:.4f} acc {test_acc:.4f} ===')
                best_epoch = epoch_idx
                best_acc = test_acc
                best_lr = args.train_learning_rates
                best_state = copy.deepcopy(model.state_dict())

    th.save(best_state, f'./parameters/{args.key}_{args.dataset_name}_{best_auc:.4f}.pt')
    logger.info(f"Training ends. The best testing auc {best_auc:.4f} acc {best_acc:.4f} at epoch {best_epoch}")
    return best_auc, best_acc, best_lr


import yaml
from data_generator_task1 import get_task1_dataloader
from data_generator_task2 import get_task2_dataloader

DATALOADER_MAP = {
    "task1": get_task1_dataloader,
    "task2": get_task2_dataloader,
}

def main():
    with open("./train_configs/train_list.yaml") as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files["files"]

    configs_list = []
    for f in file_list:
        args_list = get_args_from_yaml(f)
        configs_list += args_list

    for args in configs_list:
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info("train args")
        for k, v in args.items():
            logger.info(f"{k}: {v}")

        dataloader_manager = DATALOADER_MAP.get(args.dataset_name)

        train_loader, valid_loader, _ = dataloader_manager(
            data_path=args.dataset_name,
            batch_size=args.dataset_batch_size,
            num_workers=config.NUM_WORKER,
            edge_dropout=args.dataset_edge_dropout,
            ui=args.dataset_ui,
            us=args.dataset_us,
            si=args.dataset_si,
        )

        best_auc_acc_lr = train(args, train_loader, valid_loader, logger=logger)
        logger.info(f"best_auc_acc_lr: {best_auc_acc_lr}")


if __name__ == "__main__":
    main()

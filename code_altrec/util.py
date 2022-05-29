# -*- coding: utf-8 -*-
import numpy as np
import bottleneck as bn
import argparse, ast, math
import matplotlib.pyplot as plt

def Precision_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    precision = tmp / k
    return precision

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    # recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    recall = tmp / X_true_binary.sum(axis=1)
    return recall

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def MRR_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    topk_in_test = heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray()
    index_items = np.argwhere(topk_in_test == 1.)
    
    t = np.lexsort((index_items[:, 1], index_items[:, 0]))
    index_items[:, 0] = index_items[:, 0][t]
    index_items[:, 1] = index_items[:, 1][t]

    _, indices = np.unique(index_items[:, 0], return_index=True)
    mrr = 1. / (index_items[:, 1][indices] + 1.)
    return mrr

def parser_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--model', type=str, default='ALTRec')
    parser.add_argument('--batch_size', type=int, default=100) # keep the same as AutoRec and CDAE.
    parser.add_argument('--q_dims', nargs='+', type=int, default=None)
    parser.add_argument('--p_dims', nargs='+', type=int, default=350)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--path', type=str, default='./datasets/', help='Data path.')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Used dataset.') # ML100K", "ML1M"
    parser.add_argument('--batch_size_vad', type=int, default=5000)
    parser.add_argument('--batch_size_test', type=int, default=5000)
    parser.add_argument('--data_process', type=ast.literal_eval, default=False)
     
    parser.add_argument('--bs_g', type=int, default=100)
    parser.add_argument('--bs_d', type=int, default=100)
    parser.add_argument('--epochs_d', type=int, default=5)
    parser.add_argument('--epochs_g', type=int, default=5)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--lam_g', type=float, default=1e-2)
    
    parser.add_argument('--lam_d', type=float, default=100.)
    parser.add_argument('--adv_coeff', type=float, default=50.)
    parser.add_argument('--num_activeu', type=int, default=500)
    
    parser.add_argument('--is_valid', type=ast.literal_eval, default=True)
    parser.add_argument('--save_ckpt', type=ast.literal_eval, default=True)
    return parser.parse_args()

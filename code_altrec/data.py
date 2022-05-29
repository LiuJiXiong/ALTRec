# -*- coding: utf-8 -*-
import os
import sys
from scipy import sparse
import pandas as pd
import numpy as np


def get_count(tp, _id):
    group_by_id = tp[[_id]].groupby(_id, as_index=False)
    count = group_by_id.size()
    return count


def filter_triplets(tp, min_uc=0, min_ic=0):
    # only keep the triplets for items which were clicked on by at least min_ic users
    if min_ic > 0:
        item_count = get_count(tp, 'iid')
        tp = tp[tp['iid'].isin(item_count.index[item_count >= min_ic])]

    if min_uc > 0:
        user_count = get_count(tp, 'uid')
        tp = tp[tp['uid'].isin(user_count.index[user_count >= min_uc])]

    user_count, item_count = get_count(tp, 'uid'), get_count(tp, 'iid')
    return tp, user_count, item_count


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('uid')
    tr_list, te_list = list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['uid']))
    iid = list(map(lambda x: show2id[x], tp['iid']))
    return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])

def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1
    n_items = tp['iid'].max() + 1

    rows, cols = tp['uid'], tp['iid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data, n_items

def generate_head_tail_test(dataset, path, head_pro=10., is_vad=False):
    data_dir = os.path.join(path, dataset)
    data_dir = os.path.join(data_dir, 'weak_held_out')
    csv_file_tr = os.path.join(data_dir, 'train.csv')
    if not is_vad:
        csv_file_te = os.path.join(data_dir, 'test_te.csv')
    else:
        csv_file_te = os.path.join(data_dir, 'vad_te.csv')

    tp_tr = pd.read_csv(csv_file_tr)
    i_count = tp_tr[['iid']].groupby('iid').size()
    i_count = i_count.sort_values()
    ascending_pop_items = i_count.index

    num_clicks_arr = i_count.values
    head_items = []
    num_head_items = len(ascending_pop_items) * (head_pro/100.)
    for i in range(len(ascending_pop_items)-1, -1, -1):
        if len(head_items) < num_head_items:
            head_items.append(ascending_pop_items[i])
    print('The dataset contains %d head items and %d tail items in %.1f head proportion.' %(num_head_items, len(ascending_pop_items)-num_head_items, head_pro/100.))

    tp_te = pd.read_csv(csv_file_te)
    head_data = tp_te.loc[tp_te['iid'].isin(head_items)]
    tail_data = tp_te.loc[tp_te['iid'].isin(head_items)==False]
    if not is_vad:
        head_csv = 'head'+ str(head_pro) +'_te.csv'
        tail_csv = 'tail'+ str(head_pro) +'_te.csv'
    else:
        head_csv = 'vad_head' + str(head_pro) + '_te.csv'
        tail_csv = 'vad_tail' + str(head_pro) + '_te.csv'
    head_data.to_csv(os.path.join(data_dir, head_csv), index=False)
    tail_data.to_csv(os.path.join(data_dir, tail_csv), index=False)


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, need_transform=False):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)
    
    if need_transform:
        te_users = pd.unique(tp_te['uid'])
        tp_tr = tp_tr.loc[tp_tr['uid'].isin(te_users)]
        def continue4uid(tp, profile2id):
            uid = list(map(lambda x: profile2id[x], tp['uid']))
            iid = list(tp['iid'])
            return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])
        profile2id = dict((uid, idx) for (idx, uid) in enumerate(te_users))
        tp_tr = continue4uid(tp_tr, profile2id)
        tp_te = continue4uid(tp_te, profile2id)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['iid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['iid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

def split_tr_vad_te(data, vad_prop=0.2, te_prop=0.2):
    data_grouped_by_user = data.groupby('uid')
    tr_list, vad_list, te_list = list(), list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx1 = np.zeros(n_items_u, dtype='bool')
            idx2 = np.zeros(n_items_u, dtype='bool')
            idx3 = np.zeros(n_items_u, dtype='bool')
            index = set(range(n_items_u))
            vad_index = np.random.choice(list(index), size=int(vad_prop * n_items_u), replace=False).astype('int64')
            index = index - set(vad_index)
            te_index = np.random.choice(list(index), size=int(te_prop * n_items_u), replace=False).astype('int64')
            tr_index = np.array(list(index - set(te_index))).astype('int64')

            idx1[tr_index] = True
            idx2[vad_index] = True
            idx3[te_index] = True
            tr_list.append(group[idx1])
            vad_list.append(group[idx2])
            te_list.append(group[idx3])
        else:
            tr_list.append(group)         
    
    data_tr = pd.concat(tr_list)
    data_vad = pd.concat(vad_list)
    data_te = pd.concat(te_list)

    return data_tr, data_vad, data_te

# weak generalization
def generate_heldout_weak(dataset, path, vad_prop=0.2, te_prop=0.2):
    data_dir = os.path.join(path, dataset)
    raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), header=0)
    raw_data = raw_data[raw_data['rating']>=1] 
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %(raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    tr_triplets, vad_triplets, te_triplets = split_tr_vad_te(raw_data, vad_prop=vad_prop, te_prop=te_prop)
    tr_users = pd.unique(tr_triplets['uid'])
    vad_users = pd.unique(vad_triplets['uid'])
    te_users = pd.unique(te_triplets['uid'])
    tr_items = pd.unique(tr_triplets['iid'])

    # Keep only the triples that contain the users and items in the train set
    vad_triplets = vad_triplets.loc[vad_triplets['iid'].isin(tr_items)]
    vad_users = pd.unique(vad_triplets['uid'])
    tr4vad_triplets = tr_triplets.loc[tr_triplets['uid'].isin(vad_users)]

    te_triplets = te_triplets.loc[te_triplets['iid'].isin(tr_items)]
    te_users = pd.unique(te_triplets['uid'])
    tr4te_triplets = tr_triplets.loc[tr_triplets['uid'].isin(te_users)]

    print('Size of train/validation/test data: %d %d %d'%(len(tr_triplets), len(vad_triplets), len(te_triplets)))
    print('Size of users in train/validation/test: %d %d %d'%(len(tr_users), len(vad_users), len(te_users)))

    show2id = dict((sid, i) for (i, sid) in enumerate(tr_items))
    
    # make items close in one set.
    all_users = construct_profile2id( set(tr_users), set(vad_users), set(te_users))

    profile2id = dict((pid, i) for (i, pid) in enumerate(all_users))

    heldout_dir = os.path.join(data_dir, 'weak_held_out')

    if not os.path.exists(heldout_dir):
        os.makedirs(heldout_dir)

    with open(os.path.join(heldout_dir, 'unique_iid.txt'), 'w') as f:
        for iid in tr_items:
            f.write('%s\n' % iid)

    # Reassign new indexes to users and items (starting at 0) and save held-out data
    train_data = numerize(tr_triplets, profile2id, show2id)
    train_data.to_csv(os.path.join(heldout_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(tr4vad_triplets, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(heldout_dir, 'vad_tr.csv'), index=False)

    vad_data_te = numerize(vad_triplets, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(heldout_dir, 'vad_te.csv'), index=False)

    test_data_tr = numerize(tr4te_triplets, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(heldout_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(te_triplets, profile2id, show2id)
    test_data_te.to_csv(os.path.join(heldout_dir, 'test_te.csv'), index=False)
    print('generate held-out dataset completed\n')

# vad_users -> te_users -> tr_users
def construct_profile2id(tr_users, vad_users, te_users):
    all_users = [-1] * len(tr_users)
    # intersection between vad_users and te_users
    intersection_vad_te = vad_users & te_users
    len_intersection_vad_te = len(intersection_vad_te)
    # union between vad_users and te_users
    union_vad_te = vad_users | te_users
    len_union_vad_te = len(union_vad_te)

    idx_onlyin_vad, idx_inter_vad_te, idx_onlyin_te, idx_onlyin_tr = 0, len(vad_users) - len_intersection_vad_te, len(vad_users), len_union_vad_te
    for e in tr_users:
        # only in vad users
        if (e in vad_users) and (e not in te_users):
            all_users[idx_onlyin_vad] = e
            idx_onlyin_vad += 1
        elif e in intersection_vad_te:
            all_users[idx_inter_vad_te] = e
            idx_inter_vad_te += 1
        elif e in te_users:
            all_users[idx_onlyin_te] = e
            idx_onlyin_te += 1
        else:
            all_users[idx_onlyin_tr] = e
            idx_onlyin_tr += 1
    return all_users
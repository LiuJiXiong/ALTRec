import os, util, time, logging, shutil, random, math
from collections import defaultdict
import numpy as np
from scipy import sparse
import tensorflow as tf
import data, models

# construct pairs
def build_ids(u_active_descend, num_activeu):
    active_users = u_active_descend[0: num_activeu]
    random.shuffle(active_users)
    id1, id2, confidence = [], [], []
    length = len(u_active_descend)
    tid = 0
    for bnum, st_idx in enumerate(range(0, length, 1)):
        id1.append(u_active_descend[st_idx])
        id2.append(active_users[tid])
        confidence.append(1.)
        tid = tid + 1
        tid = tid % num_activeu

    pair =  []
    for i, (_id1, _id2, _confidence) in enumerate(zip(id1, id2, confidence)):
        pair.append([_id1, _id2, _confidence])
    pair = np.array(pair)
    return pair

def train_ALTRec(model, train_data, vad_data_tr, vad_data_te, useGPU, args, save_ckpt=False):
    if save_ckpt:
        ckpt_dir = './ckpt/ALTRec/' + args.dataset
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
    N = train_data.shape[0]
    n_click_per_user = train_data.sum() / N 
    u_clicks = zip(np.arange(N), train_data.sum(axis=1))
    u_clicks = list(u_clicks)
    u_clicks = sorted(u_clicks, key=lambda x: x[1], reverse=True)
    u_clicks = np.array(u_clicks)
    u_active_descend = list(u_clicks[:, 0])  #users in descending active order  
    batches_per_epoch = int(np.ceil(float(N) / args.batch_size))
    N_vad = vad_data_tr.shape[0]
    idxlist_vad = np.arange(N_vad)
    if useGPU:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})        
    ndcgs_vad = []    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_ndcg = -np.inf
        update_count, stop_count = 0., 0
        st_train_time, eval_time = time.clock(), 0.
        update_count = 0
        for epoch in range(args.epochs):
            for epoch_d in range(args.epochs_d):
                pair_ids = build_ids(u_active_descend=u_active_descend, num_activeu=args.num_activeu)
                loss_dis, loss_prob, loss_sim = 0., 0., 0.
                penalty = 0.
                for bnum, st_idx in enumerate(range(0, N, args.bs_d)):
                    end_idx = min(st_idx + args.bs_d, N)
                    batchIndex1 = pair_ids[:, 0][st_idx: end_idx].astype(np.int32)
                    batchIndex2 = pair_ids[:, 1][st_idx: end_idx].astype(np.int32)
                    # In fact, we do not use the confidence.
                    confidence = pair_ids[:, 2][st_idx: end_idx] 
                    X1 = train_data[batchIndex1]
                    X2 = train_data[batchIndex2]
                    if sparse.isspmatrix(X1):
                        X1 = X1.toarray()
                    if sparse.isspmatrix(X2):
                        X2 = X2.toarray()
                    X1 = X1.astype('float32')
                    X2 = X2.astype('float32')
                    feed_dict = {model.input_ph1: X1, model.input_ph2: X2, model.confidence: confidence}
                    _, _loss_dis, _loss_sim, gradients, _penalty = sess.run([model.op_dis, model.loss_dis, model.loss_sim_d, model.gradients, model.penalty], feed_dict=feed_dict)
                    loss_dis += _loss_dis
                    loss_sim += _loss_sim
                    penalty += _penalty
                logging.info('Epoch: %d D_epoch: %d loss_dis: %.4f loss_sim: %.4f penalty: %.4f' %(epoch + 1, epoch_d + 1, loss_dis, loss_sim, penalty))
            
            for epoch_g in range(args.epochs_g):                
                pair_ids = build_ids(u_active_descend=u_active_descend, num_activeu=args.num_activeu)       
                loss_gen, loss_rec, loss_sim, loss_prob = 0., 0., 0., 0.
                for bnum, st_idx in enumerate(range(0, N, args.bs_g)):
                    end_idx = min(st_idx + args.bs_g, N)
                    batchIndex1 = pair_ids[:, 0][st_idx: end_idx].astype(np.int32)
                    batchIndex2 = pair_ids[:, 1][st_idx: end_idx].astype(np.int32)
                    confidence = pair_ids[:, 2][st_idx: end_idx]
                    X1 = train_data[batchIndex1]
                    X2 = train_data[batchIndex2]
                    if sparse.isspmatrix(X1):
                        X1 = X1.toarray()
                    if sparse.isspmatrix(X2):
                        X2 = X2.toarray()
                    X1 = X1.astype('float32')
                    X2 = X2.astype('float32')
                    feed_dict = {model.input_ph1: X1, model.input_ph2: X2, model.confidence: confidence}                        
                    _, _loss_gen, _loss_rec, _loss_sim = sess.run([model.op_gen, model.loss_gen, model.loss_rec, model.loss_sim_g], feed_dict=feed_dict)
                    loss_gen += _loss_gen
                    loss_rec += _loss_rec
                    loss_sim += _loss_sim
    
                # compute validation NDCG
                ndcg_dist, st_eval_time = [], time.clock()
                for bnum, st_idx in enumerate(range(0, N_vad, args.batch_size_vad)):
                    end_idx = min(st_idx + args.batch_size_vad, N_vad)
                    batchIndex = idxlist_vad[st_idx:end_idx].astype(np.int32)
                    X = vad_data_tr[batchIndex]
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')    
                    pred_val = sess.run(model.logits1, feed_dict={model.input_ph1: X})
                    pred_val[X.nonzero()] = -np.inf
                    ndcg_dist.append(util.NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]], k=5))
                ndcg_dist = np.concatenate(ndcg_dist)
                ndcg_ = ndcg_dist.mean()
                ndcgs_vad.append(ndcg_)                  
                eval_time += time.clock() - st_eval_time
                logging.info('Epoch: %d G_Epoch: %d loss_gen: %.4f loss_rec: %.4f loss_sim: %.4f' %(epoch + 1, epoch_g + 1, loss_gen, loss_rec, loss_sim))
                logging.info('Valid_Epoch: %d Epoch_G: %d NDCG@5: %.4f'%(epoch+1, epoch_g+1, ndcg_))
                
                if ndcg_ > best_ndcg:
                    if save_ckpt: 
                        saver.save(sess, '{}/model_adv{}_numactive{}_epochs4gd{}_{}_lam4gd{}_{}_lr4gd{}_{}'.format(ckpt_dir, args.adv_coeff, args.num_activeu, args.epochs_g, args.epochs_d, args.lam_g, args.lam_d, args.lr_g, args.lr_d))
                    best_ndcg = ndcg_
                    stop_count = 0
                else:
                    stop_count += 1
                    if stop_count >= args.early_stop:
                        break  
            if stop_count >= args.early_stop:
                break
    logging.info('Best_NDCG@5:\t%.4f'%(best_ndcg))

def test_ALTRec(model, test_data_tr, test_data_te, args, flag=''):
    N_test = test_data_tr.shape[0]
    u_clicks_test = zip(np.arange(N_test), test_data_tr.sum(axis=1))
    u_clicks_test = list(u_clicks_test)
    u_clicks_test = sorted(u_clicks_test, key=lambda x: x[1])
    u_clicks_test = np.array(u_clicks_test)
    idxlist_test = u_clicks_test[:, 0]
        
    ckpt_dir = './ckpt/ALTRec/' + args.dataset
    p5_list, p20_list = [], []
    r5_list, r20_list = [], []
    n5_list, n20_list = [], []
    m5_list, m20_list = [], []
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '{}/model_adv{}_numactive{}_epochs4gd{}_{}_lam4gd{}_{}_lr4gd{}_{}'.format(ckpt_dir, args.adv_coeff, args.num_activeu, args.epochs_g, args.epochs_d, args.lam_g, args.lam_d, args.lr_g, args.lr_d))
        res = []
        for bnum, st_idx in enumerate(range(0, N_test, args.batch_size_test)):
            end_idx = min(st_idx + args.batch_size_test, N_test)
            batchIndex = idxlist_test[st_idx:end_idx].astype(np.int32)
            X = test_data_tr[batchIndex]
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
            pred_val = sess.run(model.logits1, feed_dict={model.input_ph1: X})

            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            batchIndex = idxlist_test[st_idx:end_idx].astype(np.int32)
            p5_list.append(util.Precision_at_k_batch(pred_val, test_data_te[batchIndex], k=5))
            p20_list.append(util.Precision_at_k_batch(pred_val, test_data_te[batchIndex], k=20))

            r5_list.append(util.Recall_at_k_batch(pred_val, test_data_te[batchIndex], k=5))
            r20_list.append(util.Recall_at_k_batch(pred_val, test_data_te[batchIndex], k=20))
            
            n5_list.append(util.NDCG_binary_at_k_batch(pred_val, test_data_te[batchIndex], k=5))
            n20_list.append(util.NDCG_binary_at_k_batch(pred_val, test_data_te[batchIndex], k=20))

            m5_list.append(util.MRR_at_k_batch(pred_val, test_data_te[batchIndex], k=5))
            m20_list.append(util.MRR_at_k_batch(pred_val, test_data_te[batchIndex], k=20))
        
    p5_list = np.concatenate(p5_list)
    p20_list = np.concatenate(p20_list)

    r5_list = np.concatenate(r5_list)
    r20_list = np.concatenate(r20_list)

    n5_list = np.concatenate(n5_list)
    n20_list = np.concatenate(n20_list)

    m5_list = np.concatenate(m5_list)
    m20_list = np.concatenate(m20_list)

    logging.info('Precision@5/20: %.4f \t %.4f'%(np.sum(p5_list)/N_test, np.sum(p20_list)/N_test))
    logging.info('Recall@5/20: %.4f \t %.4f '%(np.sum(r5_list)/N_test, np.sum(r20_list)/N_test))
    logging.info('NDCG@5/20: %.4f \t %.4f'%(np.sum(n5_list)/N_test, np.sum(n20_list)/N_test))
    logging.info('MRR@5/20: %.4f \t %.4f'%(np.sum(m5_list)/N_test, np.sum(m20_list)/N_test))
    
    print('Precision@5/20: %.4f \t %.4f'%(np.sum(p5_list)/N_test, np.sum(p20_list)/N_test))
    print('Recall@5/20: %.4f \t %.4f'%(np.sum(r5_list)/N_test, np.sum(r20_list)/N_test))
    print('NDCG@5/20: %.4f \t %.4f'%(np.sum(n5_list)/N_test, np.sum(n20_list)/N_test))
    print('MRR@5/20: %.4f \t %.4f'%(np.sum(m5_list)/N_test, np.sum(m20_list)/N_test))

if __name__ == "__main__":
    random.seed(123456)
    np.random.seed(123456)
    tf.set_random_seed(123456)

    args = util.parser_args()
    dataset, model, path =  args.dataset, args.model, args.path
    data_dir = os.path.join(path, dataset, 'weak_held_out')
    if args.data_process:
    	# split dataset, 60% for train, 20% for validation, the remaining for test
        data.generate_heldout_weak(dataset, path, vad_prop=0.2, te_prop=0.2)
        # we further obtain a tail data by removing the top 10% most popular items, which is used for parameter tuning.
        data.generate_head_tail_test(dataset, path, head_pro=10, is_vad=True)
        # test data by removing the top 10% and 20% most popular items in test data.
        data.generate_head_tail_test(dataset, path, head_pro=10, is_vad=False)
        data.generate_head_tail_test(dataset, path, head_pro=20, is_vad=False)
    
    if args.is_valid:
        train_data, n_items = data.load_train_data(os.path.join(data_dir, 'train.csv'))        
        vad_data_tr, vad_data_te = data.load_tr_te_data(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'vad_tail10_te.csv'), n_items, need_transform=True)
        
    else:
        train_data, n_items = data.load_train_data(os.path.join(data_dir, 'train.csv'))
        te_data_tr10, te_data_te10 = data.load_tr_te_data(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'tail10_te.csv'), n_items, need_transform=True)
        te_data_tr20, te_data_te20 = data.load_tr_te_data(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'tail20_te.csv'), n_items, need_transform=True)
    
    args.p_dims.append(n_items)
    useGPU = True
    if model == 'ALTRec':        
        log_dir = './log/' + args.model + '/' + args.dataset + '/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if args.is_valid:
            logging.basicConfig(filename=os.path.join(log_dir, "Valid_adv%.2f_numactive%d_epochs4gd%d_%d_hDim%s_lam4gd%.5f_%.5f_lr4gd%.4f_%.4f.txt" % (args.adv_coeff, args.num_activeu, args.epochs_g, args.epochs_d, str(args.p_dims), args.lam_g, args.lam_d, args.lr_g, args.lr_d)), level=logging.INFO)
        else:
            logging.basicConfig(filename=os.path.join(log_dir, "Test_adv%.2f_numactive%d_epochs4gd%d_%d_hDim%s_lam4gd%.5f_%.5f_lr4gd%.4f_%.4f.txt" % (args.adv_coeff, args.num_activeu, args.epochs_g, args.epochs_d, str(args.p_dims), args.lam_g, args.lam_d, args.lr_g, args.lr_d)), level=logging.INFO)
        logging.info(args)
        tf.reset_default_graph()
        tf.set_random_seed(123456)
        model = models.ALTRec(args)
        model.build_graph()
        if args.is_valid:
            train_ALTRec(model, train_data, vad_data_tr, vad_data_te, useGPU, args, save_ckpt=args.save_ckpt)
        else:
            tf.reset_default_graph()        
            tf.set_random_seed(123456)
            model = models.ALTRec(args)
            model.build_graph()
            logging.info('Performance on tail items with percentile 10%:')
            print('Performance on tail items with percentile 10%:')
            test_ALTRec(model, te_data_tr10, te_data_te10, args, flag='tail10')
            logging.info('Performance on tail items with percentile 20%:')
            print('\nPerformance on tail items with percentile 20%:')
            test_ALTRec(model, te_data_tr20, te_data_te20, args, flag='tail20')
    else:
        print('No model selected.')

import multiprocessing, time, sys, os

cores = 3
def exec_worker(arg):
    os.system(arg)

def run4ALTRec():
    datasets = ['ML1M', 'ML100K', 'Anime'] 
    lamgs = [1e-4, 1e-3, 1e-2, 1e-1]
    lamds = [1e-1, 1, 1e1, 1e2] 
    num_activeus = [100, 300, 500]
    adv_coeffs = [1e-2, 1e-1, 1, 10, 50, 100]
    p_dims = 350
    commands = []
    for data in datasets:
        prefix = 'python3 train.py --dataset ' + data + ' --p_dims ' + str(p_dims) 
        for lam_d in lamds:
            for lam_g in lamgs:
                for num_activeu in num_activeus:
                    for adv_coeff in adv_coeffs:
                        cmd = ' --lam_d ' + str(lam_d) + ' --lam_g ' + str(lam_g) + ' --num_activeu ' + str(num_activeu) + ' --adv_coeff ' + str(adv_coeff)
                        commands.append(prefix + cmd)
                
    pool = multiprocessing.Pool(processes=cores)
    for cmd in commands:
        pool.apply_async(exec_worker, (cmd, ))
    pool.close()
    pool.join()    

if __name__ == '__main__':
    run4ALTRec()

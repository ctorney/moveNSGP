
import threading
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

import time
import pickle


import pandas as pd
np.set_printoptions(suppress=True,precision=3)

import sys

from tensorflow_probability import distributions as tfd

from move_ns import moveNS


def setup_and_run_hmc(threadid):
    np.random.seed(threadid)
    tf.random.set_seed(threadid)

    def sp(x):
        # softplus transform with shift 
        return tf.nn.softplus(x)+1e-4

    def rbf_kernel(x1):
        # RBF kernel with single variable parameter. Other parameters are set 
        # to encode lengthscale of 20 days
        return tfp.math.psd_kernels.ExponentiatedQuadratic(x1,np.float(2.0))
        



    # initial value of kernel amplitude
    lparams_init=[0.0, 1.0] 

    aparams_init=[0.0] 

    # transform for parameter to ensure positive
    transforms=[sp] 

    # prior distribution on parameter 
    lpriors = [tfd.Normal(loc = np.float64(0.),scale=np.float64(5.)), 
               tfd.Normal(loc=np.float64(0.), scale=np.float64(10.0))]
    #              tfd.Normal(loc=np.float64(0.), scale=np.float64(10.0))]

    apriors = [tfd.Normal(loc = np.float64(0.),scale=np.float64(5.))]


    # create the model 
    mover = moveNS(T,X,Z, ID, BATCH_SIZE=1460, velocity=True,
                           #akernel=rbf_kernel, 
                           aparams_init=aparams_init, 
                           apriors=apriors, 
                           #atransforms=transforms,
                           lkernel=rbf_kernel, 
                           lparams_init=lparams_init, 
                           lpriors=lpriors, 
                           ltransforms=transforms)




    start = time.time()

    # sample from the posterior
    #mover.hmc_sample(num_samples=2000, skip=0, burn_in=1000)
    mover.hmc_sample(num_samples=4000, skip=0, burn_in=2000)
    end = time.time()
    lengths = mover.get_lengthscale_samples(X=pZ)
    np.save('data/length_switch_' + str(threadid) + '.npy',lengths)
    amps = mover.get_amplitude_samples()
    np.save('data/amp_switch_' + str(threadid) + '.npy',amps)
    print(threadid,end - start)


def parallel_run(threadid, gpu):
    with tf.name_scope(gpu):
        with tf.device(gpu):
            setup_and_run_hmc(threadid)
    return




df = pd.read_csv('data/switch.csv',index_col=0)
X = df[['Latitude','Longitude']].values
ID = df['Animal'].values

secs =(pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.seconds.astype(float).values
days = (pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.days.astype(float).values

T = (days*24*60+secs/60)/(60*24) 
T = T-T[0]
T=T[:,None]


uT = np.unique(T).copy()
z_skip=10
Z = uT[::z_skip,None].copy()
pZ = uT[::z_skip,None].copy()
np.random.shuffle(Z)

gpu_list = tf.config.experimental.list_logical_devices('GPU')
num_threads = len(gpu_list)

print(num_threads)
threads = list()
start = time.time()
for index in range(num_threads):
    x = threading.Thread(target=parallel_run, args=(index,gpu_list[index].name))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    thread.join()

end = time.time()
print('Threaded time taken: ', end-start)



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
        return tf.nn.softplus(x)#+1e-4

    def periodic_kernel(x1,x2):
        # periodic kernel with single variable parameter. Other parameters are set 
        # to encode daily activity pattern (period=1) and lengthscale of 30 minutes
        return tfp.math.psd_kernels.ExpSinSquared(x1,x2,np.float64(1.0))




    # initial value of kernel amplitude
    aparams_init=[0.0, 1.0, 0.1] 


    lparams_init=[0.0] 

    # transform for parameter to ensure positive
    atransforms=[sp, sp] 

    # prior distribution on parameter 
    apriors = [tfd.Uniform(low=np.float64(-100.), high=np.float64(100.0)),
               tfd.Normal(loc = np.float64(0.),scale=np.float64(10.)),
               tfd.Normal(loc = np.float64(0.),scale=np.float64(10.))]


    lpriors = [tfd.Uniform(low=np.float64(-100.), high=np.float64(100.0))]

    # create the model 
    mover = moveNS(T,X,Z, BATCH_SIZE=1000, 
                           lparams_init=lparams_init, 
                           lpriors=lpriors,
                           akernel=periodic_kernel, 
                           aparams_init=aparams_init, 
                           apriors=apriors, 
                           atransforms=atransforms)


    start = time.time()

    # sample from the posterior
    #mover.hmc_sample(num_samples=2000, skip=0, burn_in=1000)
    mover.hmc_sample(num_samples=2000, skip=0, burn_in=1000)
    end = time.time()

    z_len=720

    Zp = T[:z_len]# process the samples and save them for the amplitude kernel
    amps = np.power(mover.get_amplitude_samples(X=Zp),2)
    np.save('data/amps_periodic_' + str(threadid) + '.npy',amps)
    print(threadid,end - start)


def parallel_run(threadid, gpu):
    with tf.name_scope(gpu):
        with tf.device(gpu):
            setup_and_run_hmc(threadid)
    return


df = pd.read_csv('data/periodic.csv',index_col=0)
X = df[['Latitude','Longitude']].values
L = df['Lengthscale'].values
A = df['Noise'].values

secs =(pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.seconds.astype(float).values
days = (pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.days.astype(float).values

T = (days*24*60+secs/60)/(60*24) 
T = T-T[0]
T=T[:,None]


z_len=720
z_skip=18
Z = T[:z_len:z_skip]


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


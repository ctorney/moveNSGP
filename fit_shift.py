
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

    def local_periodic_kernel(x1,x2):
        # locally periodic kernel with single variable parameter. Other parameters are set 
        # to encode annual activity pattern (period=365), RBF kernel is set to allow for 
        # slow varying mean locations (2-year lengthscale).
        
        k1 = tfp.math.psd_kernels.ExpSinSquared(x1,x2,np.float64(365.0))
        k2 = tfp.math.psd_kernels.ExponentiatedQuadratic(np.float64(1.0),np.float64(2*365.0))
        #k1 = tfp.math.psd_kernels.ExpSinSquared(x1,np.float64(0.5),np.float64(365.0))
        #k2 = tfp.math.psd_kernels.ExponentiatedQuadratic(np.float64(1.0),x2*np.float64(365.0))
        #k2 = tfp.math.psd_kernels.ExponentiatedQuadratic(x2,np.float64(2*365.0))
        return k1*k2


    # initial value of kernel parameters
    mparams_init=[5.0,0.0] 
    #mparams_init=[5.0,4.0] 
    lparams_init=[0.0]
    aparams_init=[0.0]

    # prior distribution on parameters - changed to 20 
    lpriors = [tfd.Normal(loc = np.float64(0.),scale=np.float64(0.10))]
    apriors = [tfd.Normal(loc = np.float64(-1.),scale=np.float64(0.10))]

    # transform for parameter to ensure positive
    mtransforms=[sp, sp] 

    # prior distribution on parameter 
    mpriors = [tfd.Normal(loc=np.float64(0.), scale=np.float64(100.0)), tfd.Normal(loc=np.float64(0.), scale=np.float64(100.0))]

    # create the model 
    mover = moveNS(T,X,Z, BATCH_SIZE=1000, MIN_REMAIN=910,
                           mkernel=local_periodic_kernel, 
                           mparams_init=mparams_init, 
                           mpriors=mpriors, 
                           mtransforms=mtransforms,
                           aparams_init=aparams_init, 
                           apriors=apriors, 
                           lparams_init=lparams_init, 
                           lpriors=lpriors, mean_obs_noise=0, std_obs_noise=5.0 )


    start = time.time()

    # sample from the posterior
    mover.hmc_sample(num_samples=2000, skip=0, burn_in=1000)
    end = time.time()

    means_z = mover.get_mean_samples() + mean_x
    np.save('data/mean_shift_z_' + str(threadid) + '.npy',means_z)
    means = mover.get_mean_samples(X=T[::4]) + mean_x
    np.save('data/mean_shift_' + str(threadid) + '.npy',means)
    lengths = mover.get_lengthscale_samples()
    np.save('data/length_shift_' + str(threadid) + '.npy',lengths)
    amps = mover.get_amplitude_samples()
    np.save('data/amp_shift_' + str(threadid) + '.npy',amps)
    obs_noise_samples = tf.nn.softplus(mover.samples_[0]).numpy()
    np.save('data/obs_shift_' + str(threadid) + '.npy',obs_noise_samples)
    print(threadid,end - start)


def parallel_run(threadid, gpu):
    with tf.name_scope(gpu):
        with tf.device(gpu):
            setup_and_run_hmc(threadid)
    return



df = pd.read_csv('data/shift.csv',index_col=0)
X = df[['Latitude','Longitude']].values
L = df['Lengthscale'].values
meanX = df['MeanX'].values
meanY = df['MeanY'].values

mean_x = np.mean(X,axis=0)
X = X - mean_x

secs =(pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.seconds.astype(float).values
days = (pd.to_datetime(df['Date'])- pd.datetime(2000,1,1)).dt.days.astype(float).values

T = (days*24*60+secs/60)/(60*24) 
T = T-T[0]
T=T[:,None]



z_skip=4
Z = T[::z_skip].copy()
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


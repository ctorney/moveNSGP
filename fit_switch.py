
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
    lparams_init=[0.0, 3.0] 

    aparams_init=[0.0] 

    # transform for parameter to ensure positive
    transforms=[sp] 

    # prior distribution on parameter 
    lpriors = [tfd.Normal(loc = np.float64(0.),scale=np.float64(5.)), 
               tfd.Normal(loc=np.float64(3.), scale=np.float64(1))]
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
                           ltransforms=transforms, 
                           mean_obs_noise=-5, std_obs_noise=1.0)


    def build_trainable_location_scale_distribution(initial_loc, initial_scale):
        
        with tf.name_scope('build_trainable_location_scale_distribution'):
            dtype = tf.float32
            initial_loc = initial_loc * tf.ones(tf.shape(initial_scale), dtype=dtype)
            initial_scale = tf.nn.softplus(initial_scale * tf.ones_like(initial_loc))
            loc = tf.Variable(initial_value=initial_loc, name='loc')
            scale=tfp.util.TransformedVariable(tf.Variable(initial_scale, name='scale'), tfp.bijectors.Softplus())
            posterior_dist = tfd.Normal(loc=loc, scale=scale)
            posterior_dist = tfd.Independent(posterior_dist)
            
        return posterior_dist


    flat_component_dists = []

    for kparam in mover.kernel_params:
        init_loc = kparam
        init_scale = tf.random.uniform(shape=kparam.shape, minval=-2, maxval=2, dtype=tf.dtypes.float32)
        flat_component_dists.append(build_trainable_location_scale_distribution(init_loc,init_scale))

    surrogate_posterior = tfd.JointDistributionSequential(flat_component_dists)



    def target_log_prob_fn(*inputs):
        params = [tf.squeeze(a) for a in inputs]
        loss = mover.log_posterior(*params)
        return loss




    start = time.time()
    losses = tfp.vi.fit_surrogate_posterior(target_log_prob_fn, surrogate_posterior,optimizer=tf.optimizers.Adam(learning_rate=0.1, beta_2=0.9), num_steps=5000)


    steps = []
    max_step = 0.0
    
    for i in range(len(mover.kernel_params)):
        stdstep = surrogate_posterior.stddev()[i].numpy()
        meanp = surrogate_posterior.mean()[i].numpy()
        mover.kernel_params[i].assign(meanp)
        if stdstep.max()>max_step:
            max_step = stdstep.max()
        steps.append(stdstep)

    steps = [(1e-2/max_step)*s for s in steps] 

    start = time.time()

    # sample from the posterior
    num_samples=200#4000
    burn_in=500
    kr = mover.hmc_sample(num_samples=num_samples, skip=8, burn_in=burn_in, init_step=steps)
    print(np.sum(kr.inner_results.is_accepted.numpy()/num_samples))



    # sample from the posterior
    #mover.hmc_sample(num_samples=2000, skip=0, burn_in=1000)
    end = time.time()
    lengths = mover.get_lengthscale_samples(X=pZ)
    np.save('data/length_switch_' + str(threadid) + '.npy',lengths)
    amps = mover.get_amplitude_samples()
    np.save('data/amp_switch_' + str(threadid) + '.npy',amps)

    for i in range(len(mover.kernel_params)):
        output = mover.samples_[i].numpy()
        np.save('data/all_switch_' + str(i) + '_' + str(threadid) + '.npy',output)


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


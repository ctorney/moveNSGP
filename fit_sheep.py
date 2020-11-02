
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

import time
from tqdm import tqdm
import pickle

from math import *

import matplotlib.dates as mdates
import matplotlib.ticker as ticker

myFmt = mdates.DateFormatter('%Hh')

import pandas as pd
np.set_printoptions(suppress=True,precision=6)
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot') 
plt.style.use('seaborn-paper') 
plt.style.use('seaborn-whitegrid')

import sys

from tensorflow_probability import distributions as tfd
from geopy.distance import geodesic


from move_ns import moveNS



def setup_data(skip=2):
    
    df = pd.read_csv('data/ovejas.csv')

    df = df[df.id!=34]
    
    



    df['ID'] = df['id'].astype('category').cat.rename_categories(range(0, df['id'].nunique())).astype('int')
    
    ID = df['ID'].values 
    
    
    
    Xgps = df[['lat','lon']].values
    minX = np.min(Xgps[:,0])
    minY = np.min(Xgps[:,1])




    secs =(pd.to_datetime(df['time'])- pd.datetime(2018,1,1)).dt.seconds.astype(float).values
    days = (pd.to_datetime(df['time'])- pd.datetime(2018,1,1)).dt.days.astype(float).values
    T = (days*24*60+secs/60)/(60*24) #days
    T = T-np.min(T)

    rescale = 24  # use hours to improve numerical stability
    T = T * rescale


    




    # use geodesic to get the straight line distance between two points
    Xmetres = np.array([geodesic((xloc,minY), (minX,minY)).meters for xloc in Xgps[:,0]])
    Ymetres = np.array([geodesic((minX,yloc), (minX,minY)).meters for yloc in Xgps[:,1]])

    


    X = np.array([Xmetres, Ymetres]).T
    
    T=T[::skip,None]
    X=X[::skip]
    ID=ID[::skip]

    return X, T, ID


# set up positions data
X,T,ID = setup_data(skip=2)
X[:,0] = X[:,0]-X[:,0].mean()
X[:,1] = X[:,1]-X[:,1].mean()
X[:,0] = X[:,0]/1000
X[:,1] = X[:,1]/1000

# set up lower level GP locations covering 24 hours
Z = np.linspace(0,24,num=25,endpoint=False).astype(np.float64)[:,None]


def sp_shift(x):
    # softplus transform with shift 
    return tf.nn.softplus(x)+1e-4



def periodic_kernel(x1,x2):
    # periodic kernel with parameter set to encode
    # daily activity pattern (period=rescale).
    return tfp.math.psd_kernels.ExpSinSquared(x1,x2,np.float64(24.0))

# transform for parameter to ensure positive
transforms=[sp_shift,sp_shift] 
#transforms=[sp_shift] 

# diffuse priors on parameters
lpriors = [tfd.Normal(loc = np.float64(0),scale=np.float64(1)),
           tfd.Normal(loc = np.float64(0),scale=np.float64(10.))]
           
apriors = [tfd.Normal(loc = np.float64(0.),scale=np.float64(1)),
           tfd.Normal(loc = np.float64(0),scale=np.float64(10.))]


lparams_init = [0.0,0.0]
aparams_init = [0.0,0.0]


# create the model #2880
mover = moveNS(T,X,Z, ID, BATCH_SIZE=1000, MIN_REMAIN=500,velocity=True, std_obs_noise=100, mean_obs_noise=10,
                        akernel=periodic_kernel, 
                        aparams_init=aparams_init, 
                        apriors=apriors, 
                        atransforms=transforms,
                        lkernel=periodic_kernel, 
                        lparams_init=lparams_init, 
                        lpriors=lpriors, 
                        ltransforms=transforms)


#-mover.log_posterior(*mover.kernel_params)
learning_rate = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=50,
    decay_rate=0.99,
    staircase=True)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_2=0.99)
train_steps = 2000
pbar = tqdm(range(train_steps))
loss_history = np.zeros((train_steps))
for i in pbar:
    with tf.GradientTape() as t:
        loss = -mover.log_posterior(*mover.kernel_params)
    loss_history[i] = loss.numpy()
    pbar.set_description("Loss %f" % (loss_history[i]))

    gradients = t.gradient(loss, mover.kernel_params)
    optimizer.apply_gradients(zip(gradients, mover.kernel_params))
#n=3.5

opt_params = [i.numpy() for i in  mover.kernel_params]

with open('opt_params.npy', 'wb') as fp:
    pickle.dump(opt_params, fp)

with open ('opt_params.npy', 'rb') as fp:
    opt_params = pickle.load(fp)
    opt_obs_noise = opt_params[0]
    opt_ls_v = opt_params[1]
    opt_ls_amp = sp_shift(opt_params[2]).numpy()
    opt_ls_ls = sp_shift(opt_params[3]).numpy()
    opt_amp_v = opt_params[4]
    opt_amp_amp = sp_shift(opt_params[5]).numpy()
    opt_amp_ls = sp_shift(opt_params[6]).numpy()
        


start = time.time()

num_runs=4
rescale = 24



def ls_periodic_kernel():
    # periodic kernel with single variable parameter. Other parameters are set 
    # to encode daily activity pattern (period=rescale).
    # 15 minute correlation time
    return tfp.math.psd_kernels.ExpSinSquared(np.float64(opt_ls_amp),np.float64(opt_ls_ls),np.float64(24.0))

def amp_periodic_kernel():
    # periodic kernel with single variable parameter. Other parameters are set 
    # to encode daily activity pattern (period=rescale).
    # 15 minute correlation time
    return tfp.math.psd_kernels.ExpSinSquared(np.float64(opt_amp_amp),np.float64(opt_amp_ls),np.float64(24.0))


# transform for parameter to ensure positive
transforms=[] 

# prior distribution on parameters - changed to 20 
lpriors =[]#tfd.Normal(loc = np.float64(opt_ls_mean),scale=np.float64(10))]
apriors =[]#tfd.Normal(loc = np.float64(opt_amp_mean),scale=np.float64(10))]

# random initial values of mean and kernel amplitude
lparams_init =[]
aparams_init = []

# create the model 


mover_hmc = moveNS(T,X,Z, ID, BATCH_SIZE=1000, MIN_REMAIN= 500, velocity=True, std_obs_noise=100, mean_obs_noise=opt_obs_noise,
                        akernel=amp_periodic_kernel, 
                        aparams_init=aparams_init, 
                        apriors=apriors, 
                        atransforms=transforms,
                        lkernel=ls_periodic_kernel, 
                        lparams_init=lparams_init, 
                        lpriors=lpriors, 
                        ltransforms=transforms)

for i in range(num_runs):
    mover_hmc.kernel_params[0].assign(opt_obs_noise)
    mover_hmc.kernel_params[1].assign(opt_ls_v)
    mover_hmc.kernel_params[2].assign(opt_amp_v)
    
    mover_hmc.mala_sample(num_samples=500,skip=10,burn_in=20000)

    # save the results
    final_hmc_np = [j.numpy() for j in  mover_hmc.samples_]
    with open('data/hmc_samples_p_' + str(i) + '.npy', 'wb') as fp:
        pickle.dump(final_hmc_np, fp)
        
    lengths = mover_hmc.get_lengthscale_samples()
    amps = mover_hmc.get_amplitude_samples()
    np.save('data/len_sheep_p_' + str(i) + '.npy',lengths)
    np.save('data/amp_sheep_p_' + str(i) + '.npy',amps)
    

    Zin = np.linspace(0,24,num=200,endpoint=False).astype(np.float64)[:,None]

    # sample from the posterior)
    lengths = mover_hmc.get_lengthscale_samples(X=Zin)
    amps = mover_hmc.get_amplitude_samples(X=Zin)

    np.save('data/full_len_sheep_p_' + str(i) + '.npy',lengths)
    np.save('data/full_amp_sheep_p_' + str(i) + '.npy',amps)
    np.save('data/Z_pred_p_' + str(i) + '.npy',Zin)

    
    end = time.time()
    print(i,end - start)

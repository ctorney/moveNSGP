# Copyright 2019 Colin Torney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys

from tensorflow.python.eager import tape

from tensorflow_probability import distributions as tfd


class moveNS():

    def __init__(self, T, X, Z, ID=None, BATCH_SIZE=500, data_steps=[1], MIN_REMAIN=None, velocity=False,
                    lkernel=None, lparams_init=None, ltransforms=None, lpriors=None, 
                    akernel=None, aparams_init=None, atransforms=None, apriors=None, 
                    mkernel=None, mparams_init=None, mtransforms=None, mpriors=None, 
                    mean_obs_noise=0.01, std_obs_noise=10.0, **kwargs):


        self.jitter_level = 1e-6
        self.T = T 
        self.X = X
        self.Z = Z

        self.Z_ = tf.convert_to_tensor(value=Z, dtype=tf.float64)

        # a helper function to apply the custom gradient decorator to the log posterior
        self.log_posterior = tf.custom_gradient(lambda *x: moveNS._log_posterior(self, *x))

        if ID is None:
            ID = np.zeros(T.shape[0])

        if MIN_REMAIN is None:
            MIN_REMAIN=BATCH_SIZE

        self.T_ = []
        self.X_ = []
        
        for ds in data_steps:
            X_step = X[::ds,:]
            T_step = T[::ds]
            ID_step = ID[::ds]
            for j in np.unique(ID):
                XX = X_step[ID_step==j].copy()
                TT = T_step[ID_step==j].copy()
                if velocity:
                    XX = np.diff(XX,axis=0)/np.diff(TT,axis=0) # convert positions to velocities using a finite difference
                    TT = TT[:-1]+np.diff(TT,axis=0)/2 # velocity corresponds to the centre of the time interval

                for i in range(0,len(TT)//BATCH_SIZE):
                    self.T_.append(tf.convert_to_tensor(TT[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=tf.float64))
                    self.X_.append(tf.convert_to_tensor(XX[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dtype=tf.float64))

                i = len(TT)//BATCH_SIZE
                LAST_T = TT[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                LAST_X = XX[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                sz = LAST_T.shape[0]
                if sz>=MIN_REMAIN:
                    e = MIN_REMAIN*(sz//MIN_REMAIN) # only take batches that are multiple of MIN_REMAIN
                    self.T_.append(tf.convert_to_tensor(LAST_T[:e], dtype=tf.float64))
                    self.X_.append(tf.convert_to_tensor(LAST_X[:e], dtype=tf.float64))

        # measurement noise variance
        self.noise_variance = tf.Variable(mean_obs_noise, dtype=np.float32, name='obs_noise_variance')
        
        if std_obs_noise==0:
            self.noise_prior = None
            self.kernel_params = []

            self.l_start=0
            self.a_start=1
            self.m_start=2
        else:
            self.noise_prior = tfd.Normal(loc = np.float64(mean_obs_noise),scale=np.float64(std_obs_noise))
            self.kernel_params = [self.noise_variance]

            self.l_start=1
            self.a_start=2
            self.m_start=3

        # add the parameters for the lengthscale kernel
        self.kernel_params.append(tf.Variable(lparams_init[0], dtype=np.float32, name='ls_mean'))
        self.lpriors=lpriors
        if lkernel:
            self.lkernel=lkernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0]],dtype=np.float32), dtype=np.float32, name='v_ls_latents'))
            self.a_start+=1
            self.m_start+=1
            for initializer in lparams_init[1:]:
                self.kernel_params.append(tf.Variable((initializer), dtype=np.float32))
                self.a_start+=1
                self.m_start+=1
            self.ltransforms=ltransforms
            ls_params = [tf.cast(self.kernel_params[i],tf.float64) for i in range(self.l_start+2,self.a_start)]
            K_ls = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, ls_params)])

        else:
            self.lkernel=None

        # add the parameters for the amplitude kernel
        self.kernel_params.append(tf.Variable(aparams_init[0], dtype=np.float32, name='amp_mean'))
        self.apriors=apriors
        if akernel:
            self.akernel=akernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0]],dtype=np.float32), dtype=np.float32, name='v_amp_latents'))
            self.m_start+=1
            for initializer in aparams_init[1:]:
                self.kernel_params.append(tf.Variable((initializer), dtype=np.float32))
                self.m_start+=1
            self.atransforms=atransforms
            amp_params = [tf.cast(self.kernel_params[i],tf.float64) for i in range(self.a_start+2,self.m_start)]
            K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
        else:
            self.akernel=None

        # add the parameters for the mean kernel
        if mkernel:
            self.mkernel=mkernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0],2],dtype=np.float32), dtype=np.float32, name='v_mean_latents'))
            for initializer in mparams_init:
                self.kernel_params.append(tf.Variable((initializer), dtype=np.float32))
            self.mtransforms=mtransforms
            self.mpriors=mpriors
            self.rv_latents_2D = tfd.MultivariateNormalDiag(loc = tf.zeros([tf.shape(input=self.Z_)[0],2], dtype=np.float64))
            mean_params = [tf.cast(self.kernel_params[i],tf.float64) for i in range(self.m_start+1,len(self.kernel_params))]

            K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
        else:
            self.mkernel=None

        if lkernel or akernel:
            self.rv_latents = tfd.MultivariateNormalDiag(loc = tf.zeros([tf.shape(input=self.Z_)[0]], dtype=np.float64))

            
    ##########################################################################################################
    #                                                                                                        #
    #                             Code for running the HMC sampler                                           #
    #                                                                                                        #
    ##########################################################################################################
    def hmc_sample(self, num_samples=100,skip=1,burn_in=0,num_leapfrog_steps=5,init_step=1e-3):

       inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=self.log_posterior, num_leapfrog_steps=num_leapfrog_steps,step_size=init_step)#np.float32(init_step))
       kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel, num_adaptation_steps=int(burn_in * 0.8))#, adaptation_rate=0.1)#

       self.num_samples=num_samples
       samples, kernel_results = tfp.mcmc.sample_chain(num_results=self.num_samples,num_burnin_steps=burn_in,num_steps_between_results=skip,current_state=self.kernel_params, kernel=kernel)

       self.samples_ = samples
       return kernel_results

    @staticmethod
    def _log_posterior(self, *kernel_params):

        with tf.GradientTape() as grad_tape:
            for kp in kernel_params:
                grad_tape.watch(kp)

            # unpack
            ls_mean = tf.cast(kernel_params[self.l_start],tf.float64)
            amp_mean = tf.cast(kernel_params[self.a_start],tf.float64)

            if self.noise_prior is None:
                #noise_variance = tf.cast(self.noise_variance,tf.float64)
                #noise_variance = tf.cast(kernel_params[0],tf.float64)
                ret_val = self.lpriors[0].log_prob(ls_mean)
                ret_val += self.apriors[0].log_prob(amp_mean)

            else:
                noise_variance = tf.cast(kernel_params[0],tf.float64)
            
                ret_val = self.noise_prior.log_prob(noise_variance)
                ret_val += self.lpriors[0].log_prob(ls_mean)
                ret_val += self.apriors[0].log_prob(amp_mean)

            # lengthscale kernel if present
            if self.lkernel:
                ls_mean = tf.cast(kernel_params[self.l_start],tf.float64)
                ls_latents = tf.cast(kernel_params[self.l_start+1],tf.float64)
                ls_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.l_start+2,self.a_start)]

                K_ls = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, ls_params)])
                L_ls = tf.linalg.cholesky(K_ls.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)

                # ls_gp_prior = tfd.MultivariateNormalTriL(loc = ls_mean, scale_tril=L_ls).log_prob(ls_latents)
                ls_gp_prior = tfd.MultivariateNormalTriL(scale_tril=L_ls).log_prob(ls_latents)
                ret_val = ret_val + ls_gp_prior
                for p, t, a in zip(self.lpriors[1:], self.ltransforms, ls_params):
                    ret_val += p.log_prob((a))

            # amplitude kernel if present
            if self.akernel:
                amp_mean = tf.cast(kernel_params[self.a_start],tf.float64)
                amp_latents = tf.cast(kernel_params[self.a_start+1],tf.float64)
                amp_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.a_start+2,self.m_start)]

                K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
                L_amp = tf.linalg.cholesky(K_amp.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
                # amp_gp_prior = tfd.MultivariateNormalTriL(loc = amp_mean, scale_tril=L_amp).log_prob(amp_latents)
                amp_gp_prior = tfd.MultivariateNormalTriL(scale_tril=L_amp).log_prob(amp_latents)

                ret_val = ret_val + amp_gp_prior
                for p, t, a in zip(self.apriors[1:], self.atransforms, amp_params):
                    ret_val += p.log_prob((a))

            # mean kernel if present
            if self.mkernel:
                mean_latents = tf.cast(kernel_params[self.m_start],tf.float64)
                mean_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.m_start+1,len(kernel_params))]

                K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
                L_mean = tf.linalg.cholesky(K_mean.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
                    
                mean_gp_prior = tfd.MultivariateNormalTriL(scale_tril=L_mean).log_prob(mean_latents[...,0]) + \
                                    tfd.MultivariateNormalTriL(scale_tril=L_mean).log_prob(mean_latents[...,1])

                for p, t, a in zip(self.mpriors, self.mtransforms, mean_params):
                    ret_val += p.log_prob((a))

        gradients =  list(grad_tape.gradient(ret_val, kernel_params,unconnected_gradients=tf.UnconnectedGradients.ZERO))

        # tf.print(ret_val,summarize=-1,output_stream= "file:///home/staff1/ctorney/workspace/moveGP/lp11.out")
        
        with tape.stop_recording():
            for i in range(len(self.X_)):
                seg_log_prob, seg_gradients =  self.segment_log_prob(self.T_[i], self.X_[i], self.Z_, *kernel_params)
                gradients = [a + b for a, b in zip(gradients, seg_gradients)]
                ret_val = ret_val + seg_log_prob

        # tf.print(ret_val,summarize=-1,output_stream= "file:///home/staff1/ctorney/workspace/moveGP/lp22.out")
        def grad(dy):
            return [tf.cast(dy*x, tf.float32) for x in gradients]
        
        return tf.cast(ret_val,tf.float32), grad 


    @tf.function
    def segment_log_prob(self, segT, segX, Z_, *kernel_params):
        
        if self.noise_prior is None:
            noise_variance = tf.cast(self.noise_variance,tf.float64)
        else:
            noise_variance = tf.cast(kernel_params[0],tf.float64)
        ls_mean = tf.cast(kernel_params[self.l_start],tf.float64)
        amp_mean = tf.cast(kernel_params[self.a_start],tf.float64)

        if self.lkernel:
            ls_latents = tf.cast(kernel_params[self.l_start+1],tf.float64)
            ls_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.l_start+2,self.a_start)]
            K_ls = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, ls_params)])
            #L_ls = tf.linalg.cholesky(K_ls.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=np.float64) * self.jitter_level)
            #f_ls = tf.matmul((L_ls), tf.expand_dims(ls_latents,-1))

            latent_lengths = tfd.GaussianProcessRegressionModel(kernel = K_ls, index_points = segT, observations = ls_latents, observation_index_points=self.Z_).mean()
            
            
        else:
            latent_lengths = tf.zeros([tf.shape(segT)[0]], tf.float64) 


        full_lengthscales = tf.add(ls_mean,latent_lengths)
        #full_lengthscales = tf.add(ls_mean,tf.zeros_like(latent_lengths))
        length_scales = tf.nn.softplus(full_lengthscales)
        #length_scales = tf.exp(full_lengthscales)
        length_scales = tf.expand_dims(length_scales,-1)
        #tf.print(length_scales,summarize=-1,output_stream= "file:///home/staff1/ctorney/workspace/moveGP/ls.out")

        if self.akernel:
            amp_latents = tf.cast(kernel_params[self.a_start+1],tf.float64)
            amp_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.a_start+2,self.m_start)]
            K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
            #L_amp = tf.linalg.cholesky(K_amp.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
            #f_amp = tf.matmul((L_amp), tf.expand_dims(amp_latents,-1))
            latent_amplitudes = tfd.GaussianProcessRegressionModel(kernel = K_amp, index_points = segT, observations = amp_latents, observation_index_points=self.Z_).mean()
        else:
            latent_amplitudes = tf.zeros([tf.shape(segT)[0]], tf.float64) 

        full_amplitudes = tf.add(amp_mean,latent_amplitudes)
        amplitudes = tf.nn.softplus(full_amplitudes)
        #amplitudes = tf.exp(full_amplitudes)
        amplitudes = tf.expand_dims(amplitudes,-1)

        if self.mkernel:
            mean_latents = tf.cast(kernel_params[self.m_start],tf.float64)
            mean_params = [tf.cast(kernel_params[i],tf.float64) for i in range(self.m_start+1,len(kernel_params))]
            K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
            #L_mean = tf.linalg.cholesky(K_mean.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=np.float64) * self.jitter_level)
            #f_mean = tf.matmul((L_mean), mean_latents)
            latent_means_x = tfd.GaussianProcessRegressionModel(kernel = K_mean, index_points = segT, observations = mean_latents[:,0], observation_index_points=self.Z_).mean()
            latent_means_y = tfd.GaussianProcessRegressionModel(kernel = K_mean, index_points = segT, observations = mean_latents[:,1], observation_index_points=self.Z_).mean()
            latent_means = tf.stack([latent_means_x,latent_means_y])
        else:
            latent_means = tf.transpose(tf.zeros_like(segX))

        K = self.non_stat_matern12(segT, length_scales, amplitudes)
        
        K = K + (tf.eye(tf.shape(segT)[0], dtype=np.float64) * tf.nn.softplus(noise_variance))
        
        gp = tfd.MultivariateNormalTriL(loc = latent_means, scale_tril = tf.linalg.cholesky(K))

        log_prob_value = tf.reduce_sum(gp.log_prob(tf.transpose(segX)))
        gradients =  tf.gradients(log_prob_value, kernel_params)

        return log_prob_value, gradients

    def non_stat_matern12(self, X, lengthscales, stddev):

        Xs = tf.reduce_sum(input_tensor=tf.square(X), axis=-1, keepdims=True)
        Ls = tf.square(lengthscales)

        
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.linalg.matrix_transpose(Xs)
        Lscale = Ls + tf.linalg.matrix_transpose(Ls)
        dist = tf.divide(2*dist,Lscale)
        dist = tf.sqrt(tf.maximum(dist, 1e-40))
        prefactL = 2 * tf.matmul(lengthscales, lengthscales, transpose_b=True)
        prefactV = tf.matmul(stddev, stddev, transpose_b=True)

        return tf.multiply(prefactV,tf.multiply( tf.sqrt(tf.maximum(tf.divide(prefactL,Lscale), 1e-40)),tf.exp(-dist)))

    

    ##########################################################################################################
    #                                                                                                        #
    #                             Code for processing of HMC samples                                         #
    #                                                                                                        #
    ##########################################################################################################
    def get_amplitude_samples(self, batch_sz=10, X=None):

        if self.akernel:
            if X is None:
                amp_function = np.zeros((self.num_samples,np.shape(self.Z)[0],1))
            else:
                amp_function = np.zeros((self.num_samples,np.shape(X)[0],1))

            for i in range(0,self.num_samples,batch_sz):
                amp_mean_sample = self.samples_[self.a_start][i:i+batch_sz]
                amp_latents_sample = self.samples_[self.a_start+1][i:i+batch_sz]
                amp_params = []
                for j in range(len(self.atransforms)):
                    amp_params.append(self.samples_[self.a_start+2+j][i:i+batch_sz])

                amp_function[i:i+batch_sz] = self.get_amplitude_batch(amp_mean_sample, amp_latents_sample, amp_params, X)

            return amp_function
        else:
            return np.log(np.exp(self.samples_[self.a_start])+1.0)

    def get_amplitude_batch(self, amp_mean_sample, amp_latents_sample, amp_params, X=None):

        f_batch = tf.expand_dims(tf.cast(amp_latents_sample,tf.float64),-1)

        if X is not None:
            kernel_batches = self.akernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.atransforms, amp_params)])
            K_batch = kernel_batches.matrix(self.Z_,self.Z_)
            f_batch = tf.expand_dims(tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,0], observation_index_points=self.Z_).mean(),-1)
        
        f_batch = tf.nn.softplus(f_batch + tf.expand_dims(tf.expand_dims(tf.cast(amp_mean_sample,tf.float64),-1),-1))
        
        return f_batch.numpy()

    def get_lengthscale_samples(self, batch_sz = 10, X=None):
        if self.lkernel:

            if X is None:
                ls_function = np.zeros((self.num_samples,np.shape(self.Z)[0],1))
            else:
                ls_function = np.zeros((self.num_samples,np.shape(X)[0],1))

            for i in range(0,self.num_samples,batch_sz):

                ls_mean_sample = self.samples_[self.l_start][i:i+batch_sz]
                ls_latents_sample = self.samples_[self.l_start+1][i:i+batch_sz]
                ls_params = []
                for j in range(len(self.ltransforms)):
                    ls_params.append(self.samples_[self.l_start+2+j][i:i+batch_sz])

                ls_function[i:i+batch_sz] = self.get_lengthscale_batch(ls_mean_sample, ls_latents_sample, ls_params, X)

            return ls_function
        else:
            return np.log(np.exp(self.samples_[self.l_start])+1.0)

    def get_lengthscale_batch(self, ls_mean_sample, ls_latents_sample, ls_params, X=None):
    
        f_batch = tf.expand_dims(tf.cast(ls_latents_sample,tf.float64),-1)
        if X is not None:
            kernel_batches = self.lkernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.ltransforms, ls_params)])
            K_batch = kernel_batches.matrix(self.Z_,self.Z_)
            f_batch = tf.expand_dims(tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,0], observation_index_points=self.Z_).mean(),-1)
        
        f_batch = tf.nn.softplus(f_batch + tf.expand_dims(tf.expand_dims(tf.cast(ls_mean_sample,tf.float64),-1),-1))
        
        return f_batch.numpy()

    def get_mean_samples(self, batch_sz = 10, X=None):

        if self.mkernel:

            if X is None:
                mean_function = np.zeros((self.num_samples,np.shape(self.Z)[0],2))
            else:
                mean_function = np.zeros((self.num_samples,np.shape(X)[0],2))

            for i in range(0,self.num_samples,batch_sz):
                mean_latents_sample = self.samples_[self.m_start][i:i+batch_sz]
                mean_params = []
                for j in range(len(self.mtransforms)):
                    mean_params.append(self.samples_[self.m_start+1+j][i:i+batch_sz])

                mean_function[i:i+batch_sz] = self.get_mean_batch(mean_latents_sample, mean_params, X).numpy()

        return mean_function

    def get_mean_batch(self, mean_latents_sample, mean_params,X=None):

        
        f_batch = tf.cast(mean_latents_sample,tf.float64)
        if X is not None:
            kernel_batches = self.mkernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.mtransforms, mean_params)])
            K_batch = kernel_batches.matrix(self.Z_,self.Z_)
        
            f_batch_x = tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,0], observation_index_points=self.Z_).mean()
            f_batch_y = tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,1], observation_index_points=self.Z_).mean()
            f_batch = tf.stack([f_batch_x, f_batch_y],axis=-1)

        return f_batch






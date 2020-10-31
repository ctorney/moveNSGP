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

from mala_sampler import AdaptiveMALA

class moveNS():

    def __init__(self, T, X, Z, ID=None, BATCH_SIZE=500, MIN_REMAIN=None, velocity=False,
                    lkernel=None, lparams_init=None, ltransforms=None, lpriors=None, 
                    akernel=None, aparams_init=None, atransforms=None, apriors=None, 
                    mkernel=None, mparams_init=None, mtransforms=None, mpriors=None, 
                    mean_obs_noise=0.01, std_obs_noise=10.0, **kwargs):


        self.jitter_level = 1e-6
        
        self.BATCH_SIZE=BATCH_SIZE
        if MIN_REMAIN is None:
            self.MIN_REMAIN=BATCH_SIZE
        else:
            self.MIN_REMAIN=MIN_REMAIN
        self.velocity=velocity
        
        self.set_data(X,T,Z,ID)
        

        # a helper function to apply the custom gradient decorator to the log posterior
        self.log_posterior = tf.custom_gradient(lambda *x: moveNS._log_posterior(self, *x))

        # measurement noise variance
        
        if std_obs_noise==0:
            self.noise_variance = tf.constant(mean_obs_noise, dtype=tf.float64, name='obs_noise_variance')

            self.noise_prior = None
            self.kernel_params = []

            self.l_start=0
            self.a_start=1
            self.m_start=2
        else:
            self.noise_variance = tf.Variable(mean_obs_noise, dtype=tf.float64, name='obs_noise_variance')
            
            self.noise_prior = tfd.Normal(loc = np.float64(mean_obs_noise),scale=np.float64(std_obs_noise))
            self.kernel_params = [self.noise_variance]

            self.l_start=1
            self.a_start=2
            self.m_start=3

        # add the parameters for the lengthscale kernel
        self.lpriors=lpriors
        if lkernel:
            self.lkernel=lkernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0]],dtype=tf.float64), dtype=tf.float64, name='v_ls_latents'))
            for initializer in lparams_init:
                self.kernel_params.append(tf.Variable((initializer), dtype=tf.float64))
                self.a_start+=1
                self.m_start+=1
            self.ltransforms=ltransforms

        else:
            self.lkernel=None
            self.kernel_params.append(tf.Variable(lparams_init[0], dtype=tf.float64, name='ls_mean'))
        

        # add the parameters for the amplitude kernel
        self.apriors=apriors
        if akernel:
            self.akernel=akernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0]],dtype=tf.float64), dtype=tf.float64, name='v_amp_latents'))
            for initializer in aparams_init:
                self.kernel_params.append(tf.Variable((initializer), dtype=tf.float64))
                self.m_start+=1
            self.atransforms=atransforms
        else:
            self.akernel=None
            self.kernel_params.append(tf.Variable(aparams_init[0], dtype=tf.float64, name='amp_mean'))
        

        # add the parameters for the mean kernel
        if mkernel:
            self.mkernel=mkernel
            self.kernel_params.append(tf.Variable(tf.zeros([tf.shape(input=self.Z_)[0],2],dtype=tf.float64), dtype=tf.float64, name='v_mean_latents'))
            for initializer in mparams_init:
                self.kernel_params.append(tf.Variable((initializer), dtype=tf.float64))
            self.mtransforms=mtransforms
            self.mpriors=mpriors
            self.rv_latents_2D = tfd.MultivariateNormalDiag(loc = tf.zeros([tf.shape(input=self.Z_)[0],2], dtype=tf.float64))
            #mean_params = [tf.cast(self.kernel_params[i],tf.float64) for i in range(self.m_start+1,len(self.kernel_params))]

            #K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
        else:
            self.mkernel=None

        if lkernel or akernel:
            self.rv_latents = tfd.MultivariateNormalDiag(loc = tf.zeros([tf.shape(input=self.Z_)[0]], dtype=tf.float64))
    
    def set_data(self,X,T,Z,ID):
        self.Z_ = tf.convert_to_tensor(value=Z, dtype=tf.float64)
        if ID is None:
            ID = np.zeros(T.shape[0])


        self.T_ = []
        self.X_ = []
        
        for j in np.unique(ID):
            XX = X[ID==j].copy()
            TT = T[ID==j].copy()
            if self.velocity:
                XX = np.diff(XX,axis=0)/np.diff(TT,axis=0) # convert positions to velocities using a finite difference
                TT = TT[:-1]+np.diff(TT,axis=0)/2 # velocity corresponds to the centre of the time interval

            for i in range(0,len(TT)//self.BATCH_SIZE):
                self.T_.append(tf.convert_to_tensor(TT[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], dtype=tf.float64))
                self.X_.append(tf.convert_to_tensor(XX[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], dtype=tf.float64))

            i = len(TT)//self.BATCH_SIZE
            LAST_T = TT[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
            LAST_X = XX[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]

            sz = LAST_T.shape[0]
            if sz>=self.MIN_REMAIN:
                e = self.MIN_REMAIN*(sz//self.MIN_REMAIN) # only take batches that are multiple of MIN_REMAIN
                self.T_.append(tf.convert_to_tensor(LAST_T[:e], dtype=tf.float64))
                self.X_.append(tf.convert_to_tensor(LAST_X[:e], dtype=tf.float64))

        
        
            
    ##########################################################################################################
    #                                                                                                        #
    #                             Code for running the sampler                                           #
    #                                                                                                        #
    ##########################################################################################################
    def mala_sample(self, num_samples=100,skip=1,burn_in=0,init_step=1,threshold_start_estimate=500,threshold_use_estimate=1000):

        # calculate dimensions of parameter list for conversion to numpy arrays
        dims=[]
        shapes=[]
        for a in self.kernel_params:
            if tf.rank(a)==0:
                dims.append(1)
                shapes.append(1)
            else:
                dims.append(tf.size(a))
                shapes.append(tf.shape(a).numpy())
        dims=np.cumsum(dims)[:-1]

        # utility function to convert list of tf tensors to stacked numpy array
        def to_numpy(params):
            return np.hstack([mk.numpy().flatten() for mk in params])

        # utility function to set the current state 
        def set_state(array):
                
            for a, b, c in zip(np.split(array,dims), self.kernel_params, shapes):
                b.assign(np.reshape(a,c).squeeze())
            return
                
                
        # calculate the log posterior and truncated drift at x
        def log_pdf_and_drift(x):
            delta = 1000
            set_state(x)
                
            with tf.GradientTape() as t:
                logpdf = self.log_posterior(*self.kernel_params)
                
            gradients = t.gradient(logpdf, self.kernel_params)
            grad_log_pdf_x = to_numpy(gradients)
    
                
            return logpdf.numpy(), delta * grad_log_pdf_x / max(delta, np.linalg.norm(grad_log_pdf_x))

        initial_state= to_numpy(self.kernel_params)

        sampler = AdaptiveMALA(log_pdf_and_drift=log_pdf_and_drift, 
                                    state= initial_state,
                                    sigma_0=init_step,
                                    threshold_start_estimate=threshold_start_estimate,
                                    threshold_use_estimate=threshold_use_estimate)

        samples = sampler.run_sampler(num_samples,burn_in,skip)
        self.num_samples=num_samples
        self.samples_ = [tf.convert_to_tensor(sample_array.reshape(np.hstack((-1,shape))),dtype=tf.float64) for sample_array, shape in zip(np.split(samples,dims,axis=-1),shapes)]
        return 


    
    @staticmethod
    def _log_posterior(self, *kernel_params):

        with tf.GradientTape() as grad_tape:
            for kp in kernel_params:
                grad_tape.watch(kp)

            # unpack
            
            if self.noise_prior is not None:
                noise_variance = kernel_params[0]
                ret_val = self.noise_prior.log_prob(noise_variance)

            # lengthscale kernel if present
            if self.lkernel:
                ls_latents = kernel_params[self.l_start]
                #lmean = tf.reduce_mean(ls_latents)
                #ls_latents=ls_latents-lmean
                ls_params = [kernel_params[i] for i in range(self.l_start+1,self.a_start)]

                K_ls = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, ls_params)])
                L_ls = tf.linalg.cholesky(K_ls.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)

                f_ls = tf.matmul((L_ls), tf.expand_dims(ls_latents,-1))
                ls_gp_prior = tfd.MultivariateNormalTriL(loc=tf.reduce_mean(f_ls),scale_tril=L_ls).log_prob(f_ls[:,0])
               
                if self.noise_prior is None:
                    ret_val = ls_gp_prior
                else:
                    ret_val = ret_val + ls_gp_prior
                for p, t, a in zip(self.lpriors[1:], self.ltransforms, ls_params):
                    ret_val += p.log_prob((a))
            else:
                ls_mean = kernel_params[self.l_start]
                if self.noise_prior is None:
                    ret_val = self.lpriors[0].log_prob(ls_mean)
                else:
                    ret_val = ret_val + self.lpriors[0].log_prob(ls_mean)

                
            

            # amplitude kernel if present
            if self.akernel:
                #amp_mean = kernel_params[self.a_start]
                amp_latents = kernel_params[self.a_start]
                amp_params = [kernel_params[i] for i in range(self.a_start+1,self.m_start)]

                K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
                L_amp = tf.linalg.cholesky(K_amp.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
                
                f_amp = tf.matmul((L_amp), tf.expand_dims(amp_latents,-1))
                amp_gp_prior = tfd.MultivariateNormalTriL(loc=tf.reduce_mean(f_amp),scale_tril=L_amp).log_prob(f_amp[:,0])

                ret_val = ret_val + amp_gp_prior
                for p, t, a in zip(self.apriors[1:], self.atransforms, amp_params):
                    ret_val += p.log_prob((a))
            else:
                amp_mean = kernel_params[self.a_start]
                ret_val += self.apriors[0].log_prob(amp_mean)


            # mean kernel if present
            if self.mkernel:
                mean_latents = kernel_params[self.m_start]
                mean_params = [kernel_params[i] for i in range(self.m_start+1,len(kernel_params))]

                K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
                L_mean = tf.linalg.cholesky(K_mean.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
                f_mean = tf.matmul((L_mean), mean_latents)
                    
                mean_gp_prior = tfd.MultivariateNormalTriL(scale_tril=L_mean).log_prob(f_mean[...,0]) + \
                                    tfd.MultivariateNormalTriL(scale_tril=L_mean).log_prob(f_mean[...,1])
                
                ret_val = ret_val + mean_gp_prior
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
            return [dy*x for x in gradients]
        
        return ret_val, grad 

    
    @tf.function
    def segment_log_prob(self, segT, segX, Z_, *kernel_params, return_grads=True):
        
        if self.noise_prior is None:
            noise_variance = self.noise_variance
        else:
            noise_variance = kernel_params[0]
        

        if self.lkernel:
            ls_latents = kernel_params[self.l_start]
            #lmean = tf.reduce_mean(ls_latents)
            #ls_latents=ls_latents-lmean
            ls_params = [kernel_params[i] for i in range(self.l_start+1,self.a_start)]
            K_ls = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, ls_params)])
            L_ls = tf.linalg.cholesky(K_ls.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=np.float64) * self.jitter_level)
            f_ls = tf.matmul((L_ls), tf.expand_dims(ls_latents,-1))

            latent_lengths = tfd.GaussianProcessRegressionModel(kernel = K_ls,
                                                                mean_fn = lambda _: tf.reduce_mean(f_ls),
                                                                index_points = segT, 
                                                                observations = f_ls[:,0], 
                                                                observation_index_points=self.Z_).mean()
            full_lengthscales = latent_lengths#+lmean
            
        else:
            ls_mean = kernel_params[self.l_start]
            latent_lengths = tf.zeros([tf.shape(segT)[0]], tf.float64) 


            full_lengthscales = tf.add(ls_mean,latent_lengths)
        length_scales = tf.math.exp(full_lengthscales)
        length_scales = tf.expand_dims(length_scales,-1)

        if self.akernel:
            amp_latents = kernel_params[self.a_start]
            amp_params = [kernel_params[i] for i in range(self.a_start+1,self.m_start)]
            K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
            L_amp = tf.linalg.cholesky(K_amp.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
            f_amp = tf.matmul((L_amp), tf.expand_dims(amp_latents,-1))
            latent_amplitudes = tfd.GaussianProcessRegressionModel(kernel = K_amp, 
                                                                   mean_fn = lambda _: tf.reduce_mean(f_amp),
                                                                   index_points = segT, 
                                                                   observations = f_amp[:,0], 
                                                                   observation_index_points=self.Z_).mean()
            full_amplitudes = latent_amplitudes

        else:
            latent_amplitudes = tf.zeros([tf.shape(segT)[0]], tf.float64) 
            amp_mean = kernel_params[self.a_start]


            full_amplitudes = tf.add(amp_mean,latent_amplitudes)
        amplitudes = tf.math.exp(full_amplitudes)
        amplitudes = tf.expand_dims(amplitudes,-1)

        if self.mkernel:
            mean_latents = kernel_params[self.m_start]
            mean_params = [kernel_params[i] for i in range(self.m_start+1,len(kernel_params))]
            K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
            L_mean = tf.linalg.cholesky(K_mean.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=np.float64) * self.jitter_level)
            f_mean = tf.matmul((L_mean), mean_latents)
            latent_means_x = tfd.GaussianProcessRegressionModel(kernel = K_mean, 
                                                                index_points = segT, 
                                                                observations = f_mean[:,0], 
                                                                observation_index_points=self.Z_).mean()
            latent_means_y = tfd.GaussianProcessRegressionModel(kernel = K_mean, 
                                                                index_points = segT,
                                                                observations = f_mean[:,1], 
                                                                observation_index_points=self.Z_).mean()
            latent_means = tf.stack([latent_means_x,latent_means_y])
            
        else:
            latent_means = tf.transpose(tf.zeros_like(segX))

        K = self.non_stat_matern12(segT, length_scales, amplitudes)
        
        K = K + (tf.eye(tf.shape(segT)[0], dtype=tf.float64) * tf.nn.softplus(noise_variance))
        
        gp = tfd.MultivariateNormalTriL(loc = latent_means, scale_tril = tf.linalg.cholesky(K))

        log_prob_value = tf.reduce_sum(gp.log_prob(tf.transpose(segX)))
        
        if return_grads:
            gradients =  tf.gradients(log_prob_value, kernel_params)

            return log_prob_value, gradients
        
        return log_prob_value

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

    

    def get_amplitude(self, X=None):

        if self.akernel:

            amp_latents = self.kernel_params[self.a_start]
            amp_params = [self.kernel_params[i] for i in range(self.a_start+1,self.m_start)]

            K_amp = self.akernel(*[t(p) for t,p in zip(self.atransforms, amp_params)])
            L_amp = tf.linalg.cholesky(K_amp.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
            f_amp = tf.matmul((L_amp), tf.expand_dims(amp_latents,-1))
        
            if X is None:
                latent_amplitudes = f_amp
            else:
                latent_amplitudes = tfd.GaussianProcessRegressionModel(kernel = K_amp,
                                                                       mean_fn = lambda _: tf.reduce_mean(f_amp),
                                                                       index_points = X, 
                                                                       observations = f_amp[:,0],
                                                                       observation_index_points=self.Z_).mean()
            amplitudes = tf.math.exp(latent_amplitudes)
            return amplitudes.numpy()
        else:
            return np.exp(self.kernel_params[self.a_start].numpy())

    def get_lengthscale(self, X=None):

        if self.lkernel:

            len_latents = self.kernel_params[self.l_start]
            #lmean = tf.reduce_mean(len_latents)
            #len_latents=len_latents-lmean
            len_params = [self.kernel_params[i] for i in range(self.l_start+1,self.a_start)]

            K_len = self.lkernel(*[t(p) for t,p in zip(self.ltransforms, len_params)])
            L_len = tf.linalg.cholesky(K_len.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
            f_len = tf.matmul((L_len), tf.expand_dims(len_latents,-1))
        
            if X is None:
                latent_lengths = f_len# + lmean
            else:
                latent_lengths = tfd.GaussianProcessRegressionModel(kernel = K_len, 
                                                                    mean_fn = lambda _: tf.reduce_mean(f_len),
                                                                    index_points = X, 
                                                                    observations = f_len[:,0], 
                                                                    observation_index_points=self.Z_).mean()# + lmean
            lengths = tf.math.exp(latent_lengths)
            return lengths.numpy()
        else:
            return np.exp(self.kernel_params[self.l_start].numpy())

    def get_mean(self, X=None):

        if self.mkernel:

            mean_latents = self.kernel_params[self.m_start]
            mean_params = [self.kernel_params[i] for i in range(self.m_start+1,len(self.kernel_params))]
            K_mean = self.mkernel(*[t(p) for t,p in zip(self.mtransforms, mean_params)])
            L_mean = tf.linalg.cholesky(K_mean.matrix(self.Z_,self.Z_) + tf.eye(tf.shape(input=self.Z_)[0], dtype=np.float64) * self.jitter_level)
            f_mean = tf.matmul((L_mean), mean_latents)
        
            if X is None:
                latent_means = f_mean
            else:
                latent_means_x = tfd.GaussianProcessRegressionModel(kernel = K_mean, index_points = X, observations = f_mean[:,0], observation_index_points=self.Z_).mean()
                latent_means_y = tfd.GaussianProcessRegressionModel(kernel = K_mean, index_points = X, observations = f_mean[:,1], observation_index_points=self.Z_).mean()
                latent_means = tf.stack([latent_means_x,latent_means_y],axis=-1)

            return latent_means.numpy()

    ##########################################################################################################
    #                                                                                                        #
    #                             Code for processing of HMC samples                                         #
    #                                                                                                        #
    ##########################################################################################################
    def get_amplitude_samples(self, batch_sz=10, X=None):

        if self.akernel:
            if X is None:
                amp_function = np.zeros((self.num_samples,np.shape(self.Z_.numpy())[0],1))
            else:
                amp_function = np.zeros((self.num_samples,np.shape(X)[0],1))

            for i in range(0,self.num_samples,batch_sz):
                amp_latents_sample = self.samples_[self.a_start][i:i+batch_sz]
                amp_params = []
                for j in range(len(self.atransforms)):
                    amp_params.append(self.samples_[self.a_start+1+j][i:i+batch_sz,0])

                amp_function[i:i+batch_sz] = self.get_amplitude_batch(amp_latents_sample, amp_params, X)

            return amp_function
        else:
            return np.exp(self.samples_[self.a_start].numpy())

    def get_amplitude_batch(self, amp_latents_sample, amp_params, X=None):

        kernel_batches = self.akernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.atransforms, amp_params)])
        K_batch = kernel_batches.matrix(self.Z_,self.Z_)
        L_batch = (tf.linalg.cholesky(K_batch + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level))#, perm=[0,2,1])
        f_batch =  tf.matmul(tf.cast(L_batch,tf.float64), tf.expand_dims(tf.cast(amp_latents_sample,tf.float64),-1))
        if X is not None:
            f_batch = tf.expand_dims(tfd.GaussianProcessRegressionModel(kernel = kernel_batches,
                                                                        mean_fn = lambda _: tf.reduce_mean(f_batch),
                                                                        index_points = X, 
                                                                        observations = f_batch[...,0],
                                                                        observation_index_points=self.Z_).mean(),-1)
        
        f_batch = tf.math.exp(f_batch)# + tf.expand_dims(tf.expand_dims(tf.cast(amp_mean_sample,tf.float64),-1),-1))
        
        return f_batch.numpy()


    def get_lengthscale_samples(self, batch_sz = 10, X=None):
        if self.lkernel:

            if X is None:
                ls_function = np.zeros((self.num_samples,np.shape(self.Z_.numpy())[0],1))
            else:
                ls_function = np.zeros((self.num_samples,np.shape(X)[0],1))

            for i in range(0,self.num_samples,batch_sz):

                ls_latents_sample = self.samples_[self.l_start][i:i+batch_sz]
                ls_params = []
                for j in range(len(self.ltransforms)):
                    ls_params.append(self.samples_[self.l_start+1+j][i:i+batch_sz,0])

                ls_function[i:i+batch_sz] = self.get_lengthscale_batch(ls_latents_sample, ls_params, X)

            return ls_function
        else:
            return np.exp(self.samples_[self.l_start].numpy())


    def get_lengthscale_batch(self, ls_latents_sample, ls_params, X=None):

        kernel_batches = self.lkernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.ltransforms, ls_params)])
        K_batch = kernel_batches.matrix(self.Z_,self.Z_)
        L_batch = tf.linalg.cholesky(K_batch + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)
        f_batch =  tf.matmul(tf.cast(L_batch,tf.float64), tf.expand_dims(tf.cast(ls_latents_sample,tf.float64),-1))
        if X is not None:
            f_batch = tf.expand_dims(tfd.GaussianProcessRegressionModel(kernel = kernel_batches,
                                                                        mean_fn = lambda _: tf.reduce_mean(f_batch),
                                                                        index_points = X, 
                                                                        observations = f_batch[...,0], 
                                                                        observation_index_points=self.Z_).mean(),-1)
        
        f_batch = tf.math.exp(f_batch)# + tf.expand_dims(tf.expand_dims(tf.cast(ls_mean_sample,tf.float64),-1),-1))
        
        return f_batch.numpy()

    def get_mean_samples(self, batch_sz = 10, X=None):

        if self.mkernel:

            if X is None:
                mean_function = np.zeros((self.num_samples,np.shape(self.Z_.numpy())[0],2))
            else:
                mean_function = np.zeros((self.num_samples,np.shape(X)[0],2))

            for i in range(0,self.num_samples,batch_sz):
                mean_latents_sample = self.samples_[self.m_start][i:i+batch_sz]
                mean_params = []
                for j in range(len(self.mtransforms)):
                    mean_params.append(self.samples_[self.m_start+1+j][i:i+batch_sz,0])

                mean_function[i:i+batch_sz] = self.get_mean_batch(mean_latents_sample, mean_params, X).numpy()

        return mean_function


    def get_mean_batch(self, mean_latents_sample, mean_params,X=None):

        kernel_batches = self.mkernel(*[tf.cast(t(p),tf.float64) for t,p in zip(self.mtransforms, mean_params)])
        K_batch = kernel_batches.matrix(self.Z_,self.Z_)
        L_batch = tf.linalg.cholesky(K_batch + tf.eye(tf.shape(input=self.Z_)[0], dtype=tf.float64) * self.jitter_level)

        f_batch = tf.matmul(tf.cast(L_batch,tf.float64), tf.cast(mean_latents_sample,tf.float64))
        if X is not None:
            f_batch_x = tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,0], observation_index_points=self.Z_).mean()
            f_batch_y = tfd.GaussianProcessRegressionModel(kernel = kernel_batches, index_points = X, observations = f_batch[...,1], observation_index_points=self.Z_).mean()
            f_batch = tf.stack([f_batch_x, f_batch_y],axis=-1)

        return f_batch






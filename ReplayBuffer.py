#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = [] # o
        self.obs2_buf = [] # o2
        self.act_buf = np.zeros((size, 3), dtype=np.int32) # ac
        self.rew_buf = np.zeros(size, dtype=np.float32) # r
        self.done_buf = np.zeros(size, dtype=np.float32) # d
        
        self.ac_prob_buf = []
        self.log_ac_prob_buf = []
        
        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []
        self.o_embeds_buf = []

        self.ptr, self.size, self.max_size = 0, 0, size
        self.done_location = []

        # Active learning buffer
        self.sampling_buf = np.zeros(size, dtype=np.float32) # r
        self.scaler = MinMaxScaler()

        # annealing effect
        self.frame = 1
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_frames = int(1e4)

    def store(self, obs, act, rew, next_obs, done, ac_prob, log_ac_prob, \
                ac_first_prob, ac_second_hot, ac_third_prob, \
                o_embeds, sampling_score):
        if self.size == self.max_size:
            self.obs_buf.pop(0)
            self.obs2_buf.pop(0)
            
            self.ac_prob_buf.pop(0)
            self.log_ac_prob_buf.pop(0)
            
            self.ac_first_buf.pop(0)
            self.ac_second_buf.pop(0)
            self.ac_third_buf.pop(0)

            self.o_embeds_buf.pop(0)

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
        
        self.ac_prob_buf.append(ac_prob)
        self.log_ac_prob_buf.append(log_ac_prob)
        
        self.ac_first_buf.append(ac_first_prob)
        self.ac_second_buf.append(ac_second_hot)
        self.ac_third_buf.append(ac_third_prob)
        self.o_embeds_buf.append(o_embeds)
        

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.sampling_buf[self.ptr] = sampling_score

        if done:
            self.done_location.append(self.ptr)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def rew_store(self, rew, batch_size=32):
        rew_ls = list(rew)
        
        done_location_np = np.array(self.done_location)
        zeros = np.where(rew==0.0)[0]
        nonzeros = np.where(rew!=0.0)[0]
        zero_ptrs = done_location_np[zeros]        

        done_location_np = done_location_np[nonzeros]
        rew = rew[nonzeros]
        
        if len(self.done_location) > 0:
            self.rew_buf[done_location_np] += rew
            self.done_location = []

        self.act_buf = np.delete(self.act_buf, zero_ptrs, axis=0)
        self.rew_buf = np.delete(self.rew_buf, zero_ptrs)
        self.done_buf = np.delete(self.done_buf, zero_ptrs)
        self.sampling_buf = np.delete(self.sampling_buf, zero_ptrs)

        delete_multiple_element(self.obs_buf, zero_ptrs.tolist())
        delete_multiple_element(self.obs2_buf, zero_ptrs.tolist())

        delete_multiple_element(self.ac_prob_buf, zero_ptrs.tolist())
        delete_multiple_element(self.log_ac_prob_buf, zero_ptrs.tolist())
        
        delete_multiple_element(self.ac_first_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_second_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_third_buf, zero_ptrs.tolist())

        delete_multiple_element(self.o_embeds_buf, zero_ptrs.tolist())

        self.size = min(self.size-len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr-len(zero_ptrs)) % self.max_size
        
    def epistemic_var(self, x_samples, logvar_samples):
        return torch.var(x_samples, dim=1) + torch.mean(logvar_samples.exp(), dim=1)

    def sample_batch(self, device, t, batch_size=32):

        # Weighted Sampling
        sampling_score = np.nan_to_num((deepcopy(self.sampling_buf[:self.size]))**self.alpha)
        
        # Normalize importance_sampling weight
        sampling_score = self.scaler.fit_transform(sampling_score.reshape(-1, 1)) # Min-max scaler for sum to one
        sampling_score = (sampling_score/sampling_score.sum()).reshape(-1)
        
        idxs = np.random.choice([i for i in range(len(sampling_score))], 
                                size=batch_size, p=sampling_score)
        print('idxs', idxs[:20])
        print('sampling score', sampling_score.shape, sampling_score[idxs][:20])
        
        # Weighted sampling with Uncertainty calculation Every step
        
        sampling_score_batch = sampling_score[idxs]
        # Importance Correction
        beta = self.beta_by_frame(t)
        sampling_score_batch = (t*sampling_score_batch+1e-12)**(-beta)
        # normalize importance
        sampling_score_batch = self.scaler.fit_transform(sampling_score_batch.reshape(-1,1))
        sampling_score_batch = (sampling_score_batch/sampling_score_batch.sum()).reshape(-1)
        sampling_score_batch = torch.as_tensor(sampling_score_batch, dtype=torch.float32).to(device)
        print('sampling score batch', sampling_score_batch.shape, sampling_score_batch[:10])

        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        ac_prob_batch = [self.ac_prob_buf[idx] for idx in idxs]
        log_ac_prob_batch = [self.log_ac_prob_buf[idx] for idx in idxs]
        
        ac_first_batch = torch.stack([self.ac_first_buf[idx] for idx in idxs]).squeeze(1)
        ac_second_batch = torch.stack([self.ac_second_buf[idx] for idx in idxs]).squeeze(1)
        ac_third_batch = torch.stack([self.ac_third_buf[idx] for idx in idxs]).squeeze(1)
        o_g_emb_batch = torch.stack([self.o_embeds_buf[idx][2] for idx in idxs]).squeeze(1)

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch,
                     ac_prob=ac_prob_batch,
                     log_ac_prob=log_ac_prob_batch,
                     
                     ac_first=ac_first_batch,
                     ac_second=ac_second_batch,
                     ac_third=ac_third_batch,
                     o_g_emb=o_g_emb_batch,
                     idxs=idxs,
                     sampling_score=sampling_score_batch)
        return batch

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of
        learning. In practice, we linearly anneal 
        from its initial value 
        0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.sampling_buf[idx] = prio


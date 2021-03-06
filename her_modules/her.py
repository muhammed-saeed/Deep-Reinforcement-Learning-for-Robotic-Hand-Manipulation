import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        #startegy of her (final, future,episode),(random)
        self.replay_strategy = replay_strategy
        #replay_k the number of new trajectories added to the replay buffer
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            # the percentage of replay buffer taken by HER
            self.future_p = 1 - (1. / (1 + replay_k))
            print('------------------------------------')
            print('the future_p = ', self.future_p)
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        #epiosde_batch = temporary buffer --> buffer[:self.current_size]
        #batch_size _in_transition = number of episodes  in batch --> each batch (first_dim = batch_size, second_dim = max_timesteps, third_dim = len(key))

        # T maximum timesteps in the env
        T = episode_batch['actions'].shape[1]
        print('--------------------------------------')
        print('T is ', T)

        #rollout batch size =  self.current_size
        rollout_batch_size = episode_batch['actions'].shape[0]
        print('-----------------------------------------')
        print('rollout_batch_size_ ', rollout_batch_size)
        #print('self.current_size ', self.current_size)

        #batch_size = number_of_episodes in the batch
        batch_size = batch_size_in_transitions
        print('----------------------------------')
        print('bath_size ', batch_size)
        # select which rollouts and which timesteps to be used
        #np.random.randint(low = 0, high = rollout_batch_size'episode_batch['action'].shape[0]', number = batchsize)
        
        #retruns array of length batch_size and low = 0 , max = rollout_batch_size = self.current_size
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        print('--------------------------------------------------')
        print('epiosde_idxs --> (batch_size:num_episodes),(low = ),(rollout_batch_size)', episode_idxs)
        #np.random.randint(low = T, high = None, size = batch_size)
        #since high = None the values returned in range (0, low)

        #t_samples --> array its values ( from zero --> T ) and size = batch_size
        #if batch_size = 5
        #t_samples= [ 12, 49, 23,30,45] any random shape
        t_samples = np.random.randint(T, size=batch_size)
        print('---------------------------')
        print('len(t_sample', len(t_samples))
        print('t_samples, ', t_samples)
        #transitions --> episode_batch[take those episode][take those corrosponding timesteps]
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        print('------------------------------------')
        print('transitions ', transitions)
        # her idx
        her_indexes =  np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

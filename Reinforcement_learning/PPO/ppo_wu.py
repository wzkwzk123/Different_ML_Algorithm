import tensorflow as tf
import numpy as np

class PPOBuffer():
	"""
	a buffer for storing trajectories experienced by a ppo agent,
	and using Gerneralized advantage estimation(GAE-Lambda) for
	calculating the advantages of state-action pairs !!!
	"""
	def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store()

def ppo(actor_critic=ppo_algor.mlp_actor_critic, steps_per_epoch=4000):
	env = gym.make('Catrtepole_v0')
	obs_dim = env.observation_space.shape
	act_dim = env.action_space.shape

	# share information about action space with policy architecture
	ac_kwargs['action_space'] = env.action_space

	# inputs to computation graph
	state_ph action_ph = ppo_algor.placeholders_from_spaces(env.observation_space, env.action_space)
	# ?????????
	adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

	# main outputs from computation graph
	choosen_action, logp, logp_pi_choosen_action, obs_v = actor_critic(x_ph, a_ph, **ac_kwargs)

	# experience buffer
	local_steps_per_epoch = int(steps_per_epoch)
	buf = PPOBuffer()


if __name__ =='__main__':

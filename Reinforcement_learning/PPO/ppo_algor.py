import tensorflow as tf
import numpy as np

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))
    
def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def placeholder_from_spaces(space):
	if isinstance(space, Box):
		return tf.placeholder(dtype=tf.float32, shape=(None,space.shape))
	elif isinstance(space, Discrete):
		return tf.placeholder(dtype=tf.int32, shape=(None,))
	raise NotImplementedError



def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
	for h in hidden_sizes[:-1]:
		x = tf.layers.dense(x, units=h, activation=activation)
	return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)

# how to conserve the parameters of a network?
def ppo_mlp_policy(x, a , hidden_sizes, activation, output_activation, action_space):
	action_dim = action_space.n # number of  possible actions
	network_output = mlp(x, list(hidden_sizes)+[action_dim], activation, None)
	logp_all = tf.nn.log_softmax(network_output)
	choosen_action = tf.squeeze(tf.multinomial(network_output,1), axis=1)
	logp = tf.reduce_sum(tf.one_hot(a, depth=action_dim) * logp_all, axis=1)
	logp_pi_choosen_action = tf.reduce_sum(tf.one_hot(logp_pi_choosen_action, depth=action_dim) * logp_all, acis=1)
	return choosen_action, logp, logp_pi_choosen_action

def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, output_activation=None, action_space=None):
	# if policy builder depends on action space
	if policy is None and isinstance(action_space, Box):
		pass
	if policy is None and isinstance(action_space, Discrete):
		policy = ppo_mlp_policy

	with tf.variable_scope('policy'):
		choosen_action, logp, logp_pi_choosen_action = policy(x, a, hidden_sizes, activation, output_activation,action_space)
	with tf.variable_scope('v'):
		obs_v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axi=1)
	return choosen_action, logp, logp_pi_choosen_action, obs_v
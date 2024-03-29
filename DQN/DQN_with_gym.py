import tensorflow as tf
import numpy as np
import gym


tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 200
MEMORY_COUNTER = 0          # for store experience
LEARNING_STEP_COUNTER = 0   # for target updating
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory

# tf placeholders
# only occuply the memeory
tf_s = tf.placeholder(tf.float32, [None, N_STATES])
tf_a = tf.placeholder(tf.int32, [None, ])
tf_r = tf.placeholder(tf.float32, [None, ])
tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])

with tf.variable_scope('q'):        # evaluation network with 2 layers
    # dense layer  ; tf_s is the input data (state),  output (q) is the value for different actions(2)
    l_eval = tf.layers.dense(tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
    q = tf.layers.dense(l_eval, N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))

with tf.variable_scope('q_next'):   # target network, not to train
    l_target = tf.layers.dense(tf_s_, 10, tf.nn.relu, trainable=False)
    q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False) # (32 , 2)

q_target = tf_r + GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),

# the output q has the dimension of 32 4, but we only need the q value of the token action.
# a_indices  action with indices
a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
q_wrt_a = tf.gather_nd(params=q, indices=a_indices)     # shape=(None, ), q for current state

# the q_value of a state should be the value, this value are consist of two parts,
# one is the reward get from the env from the current state and the token action,
# another is the maximal q_value of the next state S_
loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))  
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def choose_action(s):

    s = s[np.newaxis, :]
    if np.random.uniform() < EPSILON:
        # forward feed the observation and get q value for every actions
        actions_value = sess.run(q, feed_dict={tf_s: s})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, N_ACTIONS)
    return action


def store_transition(s, a, r, s_):
    global MEMORY_COUNTER
    transition = np.hstack((s, [a, r], s_))
    # replace the old memory with new memory
    index = MEMORY_COUNTER % MEMORY_CAPACITY # 2000
    MEMORY[index, :] = transition
    MEMORY_COUNTER += 1


def learn():
    # update target net
    global LEARNING_STEP_COUNTER
    # merge the target net using current net
    if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
        sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    LEARNING_STEP_COUNTER += 1

    # learning
    # the return is list type
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

    b_memory = MEMORY[sample_index, :] # choose BATCH_SIZE examples by the sample_index, sample_indes is list type
    b_s = b_memory[:, :N_STATES]
    b_a = b_memory[:, N_STATES].astype(int)
    b_r = b_memory[:, N_STATES+1]
    b_s_ = b_memory[:, -N_STATES:]

    # send the 32 examples(BATCH_SIZE) to optimize
    sess.run(train_op, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset() # the shape of s is (4,) , a simple list type

    ep_r = 0
    while True:
        env.render()
        a = choose_action(s) # action is int

        # take action
        # the shape of s_ is (4,1)
        s_, r, done, info = env.step(a)
        

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        store_transition(s, a, r, s_)

        ep_r += r
        if MEMORY_COUNTER > MEMORY_CAPACITY: # after the momery is filled
            learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
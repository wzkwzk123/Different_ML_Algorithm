import tensorflow as tf
import numpy as np
class DQN():
    """docstring for DQN"""
    def __init__(self, N_ACTIONS,N_STATES):
        self.tf.set_random_seed(1)
        self.np.random.seed(1)

        # Hyper Parameters
        self.BATCH_SIZE = 32
        self.LR = 0.01                   # learning rate
        self.EPSILON = 0.9               # greedy policy
        self.GAMMA = 0.9                 # reward discount
        self.TARGET_REPLACE_ITER = 100   # target update frequency
        self.MEMORY_CAPACITY = 200
        self.MEMORY_COUNTER = 0  
        self.N_STATES = N_STATES        # for store experience
        self.N_ACTIONS = N_ACTIONS
        self.LEARNING_STEP_COUNTER = 0   # for target updating        self.N_STATES = self.env.observation_space.shape[0]
        self.MEMORY = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory

        # tf placeholders
        # only occuply the memeory
        self.tf_s = tf.placeholder(tf.float32, [None, self.N_STATES])
        self.tf_a = tf.placeholder(tf.int32, [None, ])
        self.tf_r = tf.placeholder(tf.float32, [None, ])
        self.tf_s_ = tf.placeholder(tf.float32, [None, self.N_STATES])

    def creat_net():
        
        with tf.variable_scope('q'):        # evaluation network with 2 layers
            # dense layer  ; tf_s is the input data (state),  output (q) is the value for different actions(2)
            l_eval = tf.layers.dense(self.tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            q = tf.layers.dense(l_eval, self.N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('q_next'):   # target network, not to train
            l_target = tf.layers.dense(self.tf_s_, 10, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, self.N_ACTIONS, trainable=False) # (32 , 2)

        q_target = self.tf_r + self.GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),

        # the output q has the dimension of 32 4, but we only need the q value of the token action.
        # a_indices  action with indices
        a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        q_wrt_a = tf.gather_nd(params=q, indices=a_indices)     # shape=(None, ), q for current state

        # the q_value of a state should be the value, this value are consist of two parts,
        # one is the reward get from the env from the current state and the token action,
        # another is the maximal q_value of the next state S_
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))  
        train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


    def choose_action(s): # the type of s is list

        s = s[np.newaxis, :]
        if np.random.uniform() < self.EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = sess.run(q, feed_dict={self.tf_s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action


    def store_transition(s, a, r, s_):
        global MEMORY_COUNTER
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = MEMORY_COUNTER % self.MEMORY_CAPACITY # 2000
        MEMORY[index, :] = transition
        MEMORY_COUNTER += 1


    def learn():
        # update target net
        global LEARNING_STEP_COUNTER
        # merge the target net using current net
        if self.LEARNING_STEP_COUNTER % self.TARGET_REPLACE_ITER == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        LEARNING_STEP_COUNTER += 1

        # learning
        # the return is list type
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)

        b_memory = self.MEMORY[sample_index, :] # choose BATCH_SIZE examples by the sample_index, sample_indes is list type
        b_s = b_memory[:, :self.N_STATES]
        b_a = b_memory[:, self.N_STATES].astype(int)
        b_r = b_memory[:, self.N_STATES+1]
        b_s_ = b_memory[:, -self.N_STATES:]

        # send the 32 examples(BATCH_SIZE) to optimize
        sess.run(train_op, {self.tf_s: b_s, self.tf_a: b_a, self.tf_r: b_r, self.tf_s_: b_s_})
import tensorflow as tf
import numpy as np

class ReplayMemory():

    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class DQN():
    """docstring for DQN"""
    def __init__(self, N_ACTIONS,N_STATES):
        # Hyper Parameters
        self.BATCH_SIZE = 32
        self.LR = 0.01                   # learning rate
        self.Start_EPSILON = 0.9               # greedy policy
        self.Stop_EPSILON = 0.1
        self.GAMMA = 0.9                 # reward discount
        self.Decay_rate = 0.01
        self.TARGET_REPLACE_ITER = 100   # target update frequency
        self.MEMORY_CAPACITY = 100
        self.MEMORY_COUNTER = 0  
        self.N_STATES = N_STATES        # for store experience
        self.N_ACTIONS = N_ACTIONS
        self.LEARNING_STEP_COUNTER = 0   # for target updating        self.N_STATES = self.env.observation_space.shape[0]
        self.MEMORY = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory


        self.Network = self._init_NN_DeepQ()

    def _init_NN_DeepQ(self):
           # tf placeholders
        # only occuply the memeory
        self.tf_s = tf.placeholder(tf.float32, [None, self.N_STATES])
        self.tf_a = tf.placeholder(tf.int32, [None, ])
        self.tf_r = tf.placeholder(tf.float32, [None, ])
        self.tf_s_ = tf.placeholder(tf.float32, [None, self.N_STATES])
        
        with tf.variable_scope('q'):        # evaluation network with 2 layers
            # dense layer  ; tf_s is the input data (state),  output (q) is the value for different actions(2)
            l_eval = tf.layers.dense(self.tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1),name="fc1")
            self.q = tf.layers.dense(l_eval, self.N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1),name="fc2")

        with tf.variable_scope('q_next'):   # target network, not to train
            l_target = tf.layers.dense(self.tf_s_, 10, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, self.N_ACTIONS, trainable=False) # (32 , 2)

        q_target = self.tf_r + self.GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),

        # the output q has the dimension of 32 4, but we only need the q value of the token action.
        # a_indices  action with indices
        a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)     # shape=(None, ), q for current state

        # the q_value of a state should be the value, this value are consist of two parts,
        # one is the reward get from the env from the current state and the token action,
        # another is the maximal q_value of the next state S_
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))  
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_CNN_DeepQ(self, state_size, action_size, learning_rate):
            # inputs_: [None, 84, 84, 4]
        self.inputs_ = tf.placeholder(tf.float32,
                                          [None, state_size], name="inputs")
        self.actions_ = tf.placeholder(
                tf.float32, [None, 3], name="actions")
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")

        with tf.variable_scope('q'):
            # Input: 84x84x4
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs_,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                epsilon=1e-5,
                name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                epsilon=1e-5,
                name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                epsilon=1e-5,
                name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            # --> [1152]

            # FC-layer
            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)
        #with tf.variable_scope('q_next'):
        	# not finish



            self.pred_Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            # Loss = Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.pred_Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


    def choose_action(self, s, Decay_rate_steps): # the type of s is list

        s = s[np.newaxis, :]
        Current_EPSILON = self.Stop_EPSILON + \
        (self.Start_EPSILON - self.Stop_EPSILON) * \
        np.exp(-self.Decay_rate * Decay_rate_steps)
        if np.random.uniform() < Current_EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.tf_s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.MEMORY_COUNTER % self.MEMORY_CAPACITY # 2000
        self.MEMORY[index, :] = transition
        self.MEMORY_COUNTER += 1


    def learn(self):
        # update target net
        # merge the target net using current net
        if self.LEARNING_STEP_COUNTER % self.TARGET_REPLACE_ITER == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        self.LEARNING_STEP_COUNTER += 1

        # learning
        # the return is list type
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)

        b_memory = self.MEMORY[sample_index, :] # choose BATCH_SIZE examples by the sample_index, sample_indes is list type
        b_s = b_memory[:, :self.N_STATES]
        b_a = b_memory[:, self.N_STATES].astype(int)
        b_r = b_memory[:, self.N_STATES+1]
        b_s_ = b_memory[:, -self.N_STATES:]

        # send the 32 examples(BATCH_SIZE) to optimize
        self.sess.run(self.train_op, {self.tf_s: b_s, self.tf_a: b_a, self.tf_r: b_r, self.tf_s_: b_s_})
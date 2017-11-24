



from .config import *
import tensorflow as tf

class DDPG_NET(object) :

    def __init__(self):
        self.sess = tf.Session()

        self.global_step = tf.Variable(0, trainable=False)

        self.S = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM], name='S')
        self.S_ = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM], name='S_')

        self.REWARD = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='REWARD')

        self.episode_done = tf.placeholder(dtype=tf.float32, shape=[None,1], name='Done')

        with tf.variable_scope('Actor') :
            self.a = self.build_actor_net(s = self.S, scope = 'eval', trainable=True)
            self.a_target = self.build_actor_net(s = self.S_,scope = 'target', trainable=False)

        with tf.variable_scope('Critic'):
            self.q = self.build_critic_net(s = self.S,a=self.a, scope='eval', trainable=True)
            self.q_target = self.build_critic_net(s = self.S_,a=self.a_target, scope='target', trainable=False)




        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.REWARD + GAMMA * self.q_target * (1 - self.episode_done)


        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)

        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params, global_step=self.global_step)


        tf.summary.scalar('q', tf.reduce_mean(self.q))
        tf.summary.scalar('a_loss', a_loss)
        tf.summary.scalar('q_target', tf.reduce_mean(q_target))
        tf.summary.scalar('td_error', tf.reduce_mean(td_error))
        tf.summary.scalar('reward', tf.reduce_mean(self.REWARD))
        tf.summary.histogram('a_steer', self.a[:,0])
        tf.summary.histogram('a_accel', self.a[:,1])
        tf.summary.histogram('a_brake', self.a[:,2])

        self.merged = tf.summary.merge_all()





    def build_actor_net(self, s, scope, trainable):

        with tf.variable_scope(scope):



            l1 = tf.layers.dense(inputs = s, units = L1_SIZE,
                                activation= tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable)

            l1 = tf.layers.batch_normalization(inputs=l1)

            l2 = tf.layers.dense(inputs = l1, units = L2_SIZE,
                                activation= tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable)
            l2 = tf.layers.batch_normalization(inputs=l2)

            l3_s = tf.layers.dense(l2, 100, trainable=trainable)
            l3_a = tf.layers.dense(l2, 100, trainable=trainable)
            l3_b = tf.layers.dense(l2, 100, trainable=trainable)

            steering = tf.layers.dense(inputs = l3_s, units = 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        trainable=trainable)

            acceleration = tf.layers.dense(inputs = l3_a, units = 1, activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        trainable=trainable)
            brake = tf.layers.dense(inputs = l3_b, units = 1, activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        trainable=trainable)

            a = tf.concat([steering, acceleration, brake], axis = -1)


            return a

    def build_critic_net(self, s, a, scope, trainable):

        with tf.variable_scope(scope):


            w1_s = tf.get_variable('w1_s', [INPUT_DIM, CL1_SIZE], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [ACTION_DIM, CL1_SIZE], trainable=trainable)
            b1 = tf.get_variable('b1', [1, CL1_SIZE], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net = tf.layers.batch_normalization(inputs= net)

            net = tf.layers.dense(inputs = net, units = 600, trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer()
                                  )
            net = tf.layers.batch_normalization(inputs = net)

            q = tf.layers.dense(net, 1, trainable = trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())

            return q



    def choose_action(self, session, state):
        """
        :param session: get tf session
        :param state: price tensor X
        :param pvm: portfoilo vector memory
        :return: action from poliy network
        """
        a = session.run(self.a, feed_dict = {self.S: state})






        return a[0]




    def learn(self, session, state, reward, state_, done):
        session.run(self.soft_replace)
        _ = session.run(self.atrain, feed_dict = {self.S : state})
        _, summary = session.run([self.ctrain, self.merged], 
                                 feed_dict = {self.S : state, self.S_ : state_, self.REWARD : reward, self.episode_done: done})
        return summary
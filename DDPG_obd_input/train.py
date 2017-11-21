from DDPG_obd_input.config import *
from DDPG_obd_input.gym_torcs import TorcsEnv
from DDPG_obd_input.ddpg_net import DDPG_NET
from DDPG_obd_input.utilities import Utilities
from DDPG_obd_input.memories import Memory

import tensorflow as tf
import numpy as np
# TODO save model and learn
# TODO try batch norm

class Train(object) :

    def __init__(self):

        # Define dependent lib
        self.env = TorcsEnv(vision=VISION_ON, throttle = False)
        self.net = DDPG_NET()
        self.memory = Memory()
        self.util = Utilities()

        # Initialize network
        self.train_writer = tf.summary.FileWriter('./train', self.net.sess.graph)
        self.net.sess.run(tf.global_variables_initializer())
        self.global_step = 0


    def train(self):


        for i in range(EPISODES) :

            # Reset envrionment, relaunch TORCS to prevent the memory leak error
            if i % 3 == 1 : self.state = self.env.reset(relaunch=True)
            else : self.state = self.env.reset()

            # process state
            self.state = self.util.preprocess_state(self.state)


            for j in range(MAX_STEP):

                # Exploration setting,
                epsilon = max(0, (EXPLORE - self.global_step)/EXPLORE)

                # Get Action with exploration, Ornstein-Uhlenbeck process from https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
                action = self.net.choose_action(session=self.net.sess, state=np.reshape(self.state,[-1,INPUT_DIM]))
                action = self.util.add_noise(action, epsilon)

                # Step forward
                state_, reward, done, _ = self.env.step(u=action)
                state_ = self.util.preprocess_state(state_)


                # Memory management
                self.memory.store(self.state, action, reward, state_)

                # Batch Learning
                if self.global_step > BATCH_SIZE :

                    b_state, b_action, b_reward, b_state_ = self.memory.extract_batch(BATCH_SIZE)

                    summary = self.net.learn(session=self.net.sess, state=b_state, reward=b_reward, state_=b_state_)

                    self.train_writer.add_summary(summary, self.global_step)


                # For next step
                self.state = np.copy(state_)
                self.global_step += 1


if __name__ == "__main__" :
    train = Train()
    train.train()

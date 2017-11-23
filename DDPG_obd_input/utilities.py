from .config import *
import numpy as np



class Utilities(object):

    def __init__(self):
        pass


    def Ornstein_Uhlenbeck(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)



    def add_noise(self, action, epsilon):
        action[0] = action[0] + epsilon * self.Ornstein_Uhlenbeck(x=action[0],
                                                                  theta=STEERING['theta'],
                                                                  mu=STEERING['mu'],
                                                                  sigma=STEERING['sigma'])
        action[1] = action[1] + epsilon * self.Ornstein_Uhlenbeck(x=action[1],
                                                                  theta=ACCELERATION['theta'],
                                                                  mu=ACCELERATION['mu'],
                                                                  sigma=ACCELERATION['sigma'])
        action[2] = action[2] + epsilon * self.Ornstein_Uhlenbeck(x=action[2],
                                                                  theta=BRAKE['theta'],
                                                                  mu=BRAKE['mu'],
                                                                  sigma=BRAKE['sigma'])

        return action


    def preprocess_state(self, state):

        state = np.hstack((state.angle, state.track, state.trackPos, state.speedX, state.speedY, state.speedZ, state.wheelSpinVel/100.0, state.rpm))
        #print(state)

        return state

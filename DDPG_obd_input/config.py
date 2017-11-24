# MODEL Hyperparameter

# For training network
INPUT_DIM = 29
ACTION_DIM = 3
BATCH_SIZE = 32
MEMORY_SIZE = 100000
EPISODES = 10000
MAX_STEP = 100000
GAMMA = 0.95
LR_C = 1e-5
LR_A = 1e-6
TAU = 0.001
L2 = 0.01
REGULIZE_COEFF = 1e-5
L1_SIZE = 300
L2_SIZE = 600
CL1_SIZE = 300
CL2_SIZE = 600

EXPLORE = 100000
START_EPSILON = 1
FINAL_EPSILON = 0.05


# For Tensorflow
LOAD_MODEL = True
SAVE_TERM = 10000


# Ornstein Uhlenbeck process
# Referred constants from https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
STEERING = {'theta' :0.9, 'mu' : 0.0, 'sigma' : 0.2}
ACCELERATION = {'theta' : 1.0, 'mu' : 0.6, 'sigma' : 0.1}
BRAKE = {'theta' : 1.0, 'mu' : -0.1, 'sigma' : 0.05}



VISION_ON =False

# Trainig Options
TRAINING = True


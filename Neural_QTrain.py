import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100   # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.99               # discount factor          
INITIAL_EPSILON = 0.75       # starting value of epsilon
FINAL_EPSILON = 0.05         # final value of epsilon
EPSILON_DECAY_STEPS = 100    # decay period
HIDDEN_NODES = 128           # hidden nodes in network graph
L2_REG_BETA = 0.001          # L2 regularisation coefficient

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# Create a Double DQN with Experience Replay

# Online Network
W1_on = tf.Variable(tf.truncated_normal([STATE_DIM, HIDDEN_NODES]), name='W1_on')
b1_on = tf.Variable(tf.constant(0., shape=[HIDDEN_NODES]), name='b1_on')
HL_1_on = tf.nn.relu(tf.matmul(state_in, W1_on) + b1_on, name='HL_1_on')

W2_on = tf.Variable(tf.truncated_normal([HIDDEN_NODES, HIDDEN_NODES]), name='W2_on')
b2_on = tf.Variable(tf.constant(0., shape=[HIDDEN_NODES]), name='b2_on')
HL_2_on = tf.nn.relu(tf.matmul(HL_1_on, W2_on) + b2_on, name='HL_2_on')

W3_on = tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2]), name='W3_on')
b3_on = tf.Variable(tf.constant(0., shape=[2]), name='b3_on')
q_values = tf.matmul(HL_2_on, W3_on) + b3_on

# Target Network
W1_tn = tf.Variable(tf.truncated_normal([STATE_DIM, HIDDEN_NODES]), name='W1_tn', trainable=False)
b1_tn = tf.Variable(tf.constant(0., shape=[HIDDEN_NODES]), name='b1_tn', trainable=False)
HL_1_tn = tf.nn.relu(tf.matmul(state_in, W1_tn) + b1_tn, name='HL_1_tn')

W2_tn = tf.Variable(tf.truncated_normal([HIDDEN_NODES, HIDDEN_NODES]), name='W2_tn', trainable=False)
b2_tn = tf.Variable(tf.constant(0., shape=[HIDDEN_NODES]), name='b2_tn', trainable=False)
HL_2_tn = tf.nn.relu(tf.matmul(HL_1_tn, W2_tn) + b2_tn, name='HL_2_tn')

W3_tn = tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2]), name='W3_tn', trainable=False)
b3_tn = tf.Variable(tf.constant(0., shape=[2]), name='b3_tn', trainable=False)
target_q_values = tf.matmul(HL_2_tn, W3_tn) + b3_tn

# Network outputs
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
l2 = L2_REG_BETA*sum(tf.nn.l2_loss(tf_var)
                        for tf_var in tf.trainable_variables() 
                            if not ("b" in tf_var.name))
loss = tf.reduce_mean(tf.square(target_in - q_action), name='loss') + l2
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

# parameters and variables to implement experience memory "replay buffer"
# ideally should be a deque but we can't import one so a simple version
# based on a list and an insert using an index 
# popping is inefficient - O(n) at each step where n is the size of the memory
BATCH_SIZE = 256    
MEMORY_SIZE = 10000
memory = [] 
mem_idx = 0

def train():

    training_size = min(len(memory), BATCH_SIZE)
    minibatch = random.sample(memory, training_size)
    states = [data[0] for data in minibatch]
    actions = [data[1] for data in minibatch]
    rewards = [data[2] for data in minibatch]
    next_states = [data[3] for data in minibatch]
    is_dones = [data[4] for data in minibatch]
    
    Q_next_online = q_values.eval(feed_dict={state_in: next_states})
    Q_next_train = target_q_values.eval(feed_dict={state_in: next_states})
    Q_values = [Q_next_train[i][np.argmax(Q_next_online, axis=1)[i]] for i in range(training_size)]
    
    targets = []
    for i in range(training_size):
        if is_dones[i]:
            targets.append(rewards[i])
        else:
            targets.append(rewards[i] + GAMMA*Q_values[i])
    session.run([optimizer], feed_dict={
        target_in: targets,
        action_in: actions,
        state_in: states
    })

def remember(state, action, reward, next_state, done):
    global mem_idx, memory

    if len(memory) == MEMORY_SIZE:
        if mem_idx == MEMORY_SIZE:
            mem_idx = 0
        memory[mem_idx] = (state, action, reward, next_state, done) 
        mem_idx += 1
    else:
        memory.append((state, action, reward, next_state, done))

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS
    
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        # memorize
        remember(state, action, reward, next_state, done)

        # get a batch and train
        train()

        # Update
        state = next_state

        if done:
            break

        # if step == 100:
        #     session.run([W1_tn.assign(W1_on), 
        #         b1_tn.assign(b1_on),
        #         W2_tn.assign(W2_on),
        #         b2_tn.assign(b2_on),
        #         W3_tn.assign(W3_on),
        #         b3_tn.assign(b3_on)])

    session.run([W1_tn.assign(W1_on), 
                b1_tn.assign(b1_on),
                W2_tn.assign(W2_on),
                b2_tn.assign(b2_on),
                W3_tn.assign(W3_on),
                b3_tn.assign(b3_on)])
    
    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()

import numpy as np
import random
import tensorflow as tf
import model_tf
import os
import visulization
import controller
import pygame
import skimage.io
import skimage.transform

INIT_EP = 1
FINAL_EP = 0.1
BATCH_SIZE = 32
NB_FRAMES_TO_STOP_EXPLORING = 1000
REPLAY_MEMORY_LIMIT = 50000
UPDATE_FREQ = 4
ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "suck": 4
}
INDICES_TO_ACTIONS = {v: k for k, v in ACTIONS.iteritems()}
GAMMA = .99  # Discount factor on the target Q-values
NUM_EPISODES = 1000  # How many episodes of game environment to train network with.
NB_OBSERVATIONS = 500  # How many steps of random actions before training begins.
MAX_EP_LENGTH = 5000  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
path = "./models_ddqn"  # The path to save our model to.
TAU = 0.001  # Rate to update target network toward primary network


class Experience_Buffer():
    def __init__(self, buffer_size=REPLAY_MEMORY_LIMIT):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(
            tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, _sess):
    for op in op_holder:
        _sess.run(op)


def get_next_frame(action=None):
    if action is not None:
        CONTROLLER.control(action)
    GAME.draw()
    img = pygame.surfarray.array3d(GAME.get_surface())
    skimage.io.imsave("frames/frame1.jpg", img)
    img = skimage.transform.resize(img, (84, 84, 3))
    _s = (CONTROLLER.robot.get_tile_pos()[0], CONTROLLER.robot.get_tile_pos()[1], tuple(CONTROLLER.dirty_tiles))
    return img, CONTROLLER.reward_func(_s), CONTROLLER.terminate()


tf.reset_default_graph()
mainQN = model_tf.Q_network()
targetQN = model_tf.Q_network()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, TAU)

myBuffer = Experience_Buffer()

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(NUM_EPISODES):
        # Set the rate of random action decrease.
        e = INIT_EP
        stepDrop = (INIT_EP - FINAL_EP) / NB_FRAMES_TO_STOP_EXPLORING

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0
        GAME = visulization.GUI()
        CONTROLLER = controller.ManualController(GAME.get_robot())
        loss = None
        episode_buffer = Experience_Buffer()

        # Reset environment and get first new observation
        s0 = get_next_frame()[0]
        s0 = np.reshape(s0, [21168])
        d = False
        rAll = 0
        j = 0
        print "Training episode %d" % i
        while j < MAX_EP_LENGTH:
            j += 1
            if np.random.rand(1) < e or total_steps < NB_OBSERVATIONS:
                a = np.random.randint(0, len(ACTIONS))
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s0]})[0]

            s1, r, d = get_next_frame(INDICES_TO_ACTIONS[a])
            s1 = np.reshape(s1, [21168])
            total_steps += 1
            episode_buffer.add(np.reshape(np.array([s0, a, r, s1, d]), [1, 5]))

            if total_steps > NB_OBSERVATIONS:
                if e > FINAL_EP:
                    e -= stepDrop
                if total_steps % UPDATE_FREQ == 0:
                    try:
                        trainBatch = myBuffer.sample(BATCH_SIZE)
                    except ValueError:
                        trainBatch = episode_buffer.sample(BATCH_SIZE)

                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Q_out, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = 1 - trainBatch[:, 4]
                    doubleQ = Q2[range(BATCH_SIZE), Q1]
                    targetQ = trainBatch[:, 2] + (GAMMA * doubleQ * end_multiplier)

                    # Update the network with our target values.
                    _, loss = sess.run([mainQN.updateModel, mainQN.loss],
                                       feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                  mainQN.targetQ: targetQ,
                                                  mainQN.actions: trainBatch[:, 1]})
                    print "Loss is %f" % (loss)
                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s0 = s1

            if d:
                break

        myBuffer.add(episode_buffer.buffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if i % 100 == 0:
            saver.save(sess, path + '/model_for_test' + '.ckpt')
            print "Saved Model"
        if len(rList) % 10 == 0:
            print total_steps, np.mean(rList[-10:]), e
    saver.save(sess, path + '/model-main.ckpt')
print "Percent of succesful episodes: " + str(sum(rList) / NUM_EPISODES)

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
import curses
import pickle

INIT_EP = 1
FINAL_EP = 0.1
BATCH_SIZE = 32
NB_FRAMES_TO_STOP_EXPLORING = 1000
REPLAY_MEMORY_LIMIT = 50000
UPDATE_FREQ = 4  # Rate to train the model
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
STDSRC = curses.initscr()


class ExperienceBuffer:
    """
    Buffer to store states
    """
    def __init__(self, buffer_size=REPLAY_MEMORY_LIMIT):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def update_target_graph(tfVars, tau):
    """
    Update target graph
    :param tfVars:
    :param tau:
    :return: list of op
    """
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(
            tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def update_target(op_holder, _sess):
    """
    Run the op to update target graph
    :param op_holder:
    :param _sess:
    :return:
    """
    for op in op_holder:
        _sess.run(op)


def get_next_frame(action=None):
    """
    Return the next frame
    :param action: action to be performed
    :return: next frame, reward, terminated or not
    """
    if action is not None:
        CONTROLLER.control(action)
    GAME.draw()
    img = pygame.surfarray.array3d(GAME.get_surface())
    img = skimage.transform.resize(img, (84, 84, 3))
    _s = (CONTROLLER.robot.get_tile_pos()[0], CONTROLLER.robot.get_tile_pos()[1], tuple(CONTROLLER.dirty_tiles))
    return img, CONTROLLER.reward_func(_s), CONTROLLER.terminate()


def print_log_string(log_string, row, col=0):
    """
    Print the log string from curses
    :param log_string:
    :param row:
    :param col:
    :return:
    """
    STDSRC.addstr(row, col, log_string)


tf.reset_default_graph()
mainQN = model_tf.DoubleQnetwork()
targetQN = model_tf.DoubleQnetwork()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = update_target_graph(trainables, TAU)

myBuffer = ExperienceBuffer()

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    nb_iterations_list = []
    rewards_list = []
    avg_losses_list = []
    for i in range(NUM_EPISODES):
        curses.noecho()
        curses.cbreak()
        # set epsilon
        e = INIT_EP
        stepDrop = (INIT_EP - FINAL_EP) / NB_FRAMES_TO_STOP_EXPLORING

        # reset env
        GAME = visulization.GUI()
        CONTROLLER = controller.ManualController(GAME.get_robot())
        max_reward = len(CONTROLLER.dirty_tiles) * 0.01
        episode_buffer = ExperienceBuffer()
        s0 = get_next_frame()[0]
        s0 = np.reshape(s0, [21168])
        d = False
        total_reward = 0
        nb_iters = 0
        total_loss = 0
        exploited = False

        while nb_iters < MAX_EP_LENGTH:
            print_log_string("Ep: %d, Iter: %d, Max reward: %f\n" % (i, nb_iters, max_reward), 1)
            nb_iters += 1
            if np.random.rand(1) < e or nb_iters < NB_OBSERVATIONS:
                print_log_string("explore epsilon = %f\n" % e, 2, col=2)
                a = np.random.randint(0, len(ACTIONS))
            else:
                print_log_string("exploit epsilon = %f\n" % e, 2, col=2)
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s0]})[0]
                exploited = True

            s1, r, d = get_next_frame(INDICES_TO_ACTIONS[a])
            s1 = np.reshape(s1, [21168])
            episode_buffer.add(np.reshape(np.array([s0, a, r, s1, d]), [1, 5]))

            print_log_string("executing action %s\n" % INDICES_TO_ACTIONS[a], 3, col=2)
            print_log_string("get reward %f\n" % r, 4, col=2)

            if nb_iters > NB_OBSERVATIONS:
                if e > FINAL_EP:
                    e -= stepDrop
                if nb_iters % UPDATE_FREQ == 0:
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
                    _, loss_val = sess.run([mainQN.updateModel, mainQN.loss],
                                           feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                      mainQN.targetQ: targetQ,
                                                      mainQN.actions: trainBatch[:, 1]})
                    total_loss += loss_val
                    print_log_string("loss is %f, overall loss is %f\n" %
                                     (loss_val, total_loss / nb_iters), 6, col=2)
                    print_log_string("total reward acquired is %f\n" % total_reward, 7, col=2)
                    update_target(targetOps, sess)  # Update the target network toward the primary network.
            if exploited:
                total_reward += r
                exploited = False
            s0 = s1
            STDSRC.refresh()
            curses.echo()
            curses.nocbreak()

            # if terminated
            if d:
                break

        myBuffer.add(episode_buffer.buffer)

        # add stats
        nb_iterations_list.append(nb_iters)
        rewards_list.append(total_reward)
        avg_losses_list.append(total_loss/nb_iters)

        # Periodically save the model.
        if i % 100 == 0:
            saver.save(sess, path + '/model-main' + '.ckpt')

    # save the model and stats
    saver.save(sess, path + '/model-main.ckpt')
    with open('stats/avg_reward1' + '.pkl', 'wb') as f:
        pickle.dump(rewards_list, f, pickle.HIGHEST_PROTOCOL)
    with open('stats/avg_loss1' + '.pkl', 'wb') as f:
        pickle.dump(avg_losses_list, f, pickle.HIGHEST_PROTOCOL)
    with open('stats/ep_length1' + '.pkl', 'wb') as f:
        pickle.dump(nb_iterations_list, f, pickle.HIGHEST_PROTOCOL)
    curses.endwin()

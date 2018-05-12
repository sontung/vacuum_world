from __future__ import print_function
import tensorflow as tf
import sys
import numpy as np
import pygame
import controller
import random
import visulization
import skimage.io
import skimage.color
import skimage.exposure
import skimage.transform
import collections
import model_tf
import curses
import pickle


N = 50000
M = 50  # nb episodes
C = 5000  # target network update freq
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
INIT_EP = 1
FINAL_EP = 0.1
ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "suck": 4
}
INDICES_TO_ACTIONS = {v: k for k, v in ACTIONS.iteritems()}
NB_OBSERVATIONS = 256
NB_FRAMES_TO_STOP_EXPLORING = 5000
GAMMA = 0.99
REPLAY_MEMORY_LIMIT = 50000
MAX_EP_LENGTH = 5000
path = "./models_ddqn"  # The path to save our model to.


def print_log_string(log_string, row, col=0):
    STDSRC.addstr(row, col, log_string)


def update_target_net(sess):
    """
    Update action net to target net
    """
    action_vars = tf.trainable_variables(scope="action")
    target_vars = tf.trainable_variables(scope="target")
    for i in range(len(action_vars)):
        sess.run(target_vars[i].assign(action_vars[i].value()))


def get_next_frame(action=None):
    global GAME, CONTROLLER
    if action is not None:
        CONTROLLER.control(action)
    GAME.draw()
    _img = pygame.surfarray.array3d(GAME.get_surface())
    _img = np.array(_img)
    _img = skimage.color.rgb2gray(_img)
    _img = skimage.transform.resize(_img, (84, 84))
    _img = skimage.exposure.rescale_intensity(_img, out_range=(0, 255))
    s = (CONTROLLER.robot.get_tile_pos()[0], CONTROLLER.robot.get_tile_pos()[1], tuple(CONTROLLER.dirty_tiles))
    return _img, CONTROLLER.reward_func(s), CONTROLLER.terminate()


graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope("action"):
        action_net = model_tf.deepQnet(lr=LEARNING_RATE, nb_actions=5)
    with tf.variable_scope("target"):
        target_net = model_tf.deepQnet(lr=LEARNING_RATE, nb_actions=5)
    saver = tf.train.Saver()


if __name__ == "__main__":
    STDSRC = curses.initscr()
    with tf.Session(graph=graph) as session:
        replay_memory = collections.deque()
        tf.global_variables_initializer().run()
        update_target_net(session)
        step = 0
        avg_reward_list = []
        total_reward_list = []
        ep_length = []
        for episode in range(M):
            curses.noecho()
            curses.cbreak()
            GAME = visulization.GUI()
            CONTROLLER = controller.ManualController(GAME.get_robot())
            img, reward, terminate = get_next_frame()
            sample = np.stack((img, img, img, img), axis=2)
            sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
            epsilon = INIT_EP
            nb_iterations = 0
            loss = 0
            total_reward = 0
            exploited = False
            while not CONTROLLER.terminate() and nb_iterations <= MAX_EP_LENGTH:
                print_log_string("Ep: %d, Iter: %d:\n" % (episode, nb_iterations), 1)

                # Choose action based on epsilon
                action = np.zeros(len(ACTIONS))
                if random.random() <= epsilon:
                    print_log_string("explore epsilon = %f\n" % epsilon, 2, col=2)
                    index = random.choice(ACTIONS.values())
                    action[index] = 1.0
                else:
                    print_log_string("exploit epsilon = %f\n" % epsilon, 2, col=2)
                    q_values = session.run([action_net.output], feed_dict={action_net.input: sample})
                    index = np.argmax(q_values)
                    action[index] = 1.0
                    exploited = True

                # Decrease epsilon
                if epsilon > FINAL_EP and nb_iterations > NB_OBSERVATIONS:
                    epsilon -= (INIT_EP - FINAL_EP) / NB_FRAMES_TO_STOP_EXPLORING

                # Execute action and get next frame
                action_in_string = INDICES_TO_ACTIONS[index]
                img2, reward2, terminate2 = get_next_frame(action_in_string)
                img2 = img2.reshape(1, img2.shape[0], img2.shape[1], 1)  # 1x84x84x1
                sample2 = np.append(img2, sample[:, :, :, :3], axis=3)
                if exploited:
                    total_reward += reward2
                    exploited = False

                print_log_string("executing action %s\n" % action_in_string, 3, col=2)
                print_log_string("get reward %f\n" % reward2, 4, col=2)

                # Store transition in replay memory
                replay_memory.append((sample, index, reward2, sample2, terminate2))
                print_log_string("replay memory has %d samples\n" % len(replay_memory), 5, col=2)
                if len(replay_memory) > REPLAY_MEMORY_LIMIT:
                    replay_memory.popleft()

                # Train when observe enough
                if nb_iterations > NB_OBSERVATIONS:
                    mini_batch = random.sample(replay_memory, BATCH_SIZE)
                    inputs = np.zeros((BATCH_SIZE, 84, 84, 4))
                    targets = np.zeros((BATCH_SIZE, len(ACTIONS)))
                    target_actions = np.zeros(BATCH_SIZE)
                    for i in range(0, len(mini_batch)):
                        state_t = mini_batch[i][0]
                        action_t = mini_batch[i][1]
                        reward_t = mini_batch[i][2]
                        state_t1 = mini_batch[i][3]
                        terminal = mini_batch[i][4]

                        inputs[i] = state_t
                        target_actions[i] = action_t

                        q_sa = session.run([target_net.output], feed_dict={target_net.input: state_t1})

                        if terminal:
                            targets[i, action_t] = reward_t
                        else:
                            targets[i, action_t] = reward_t + GAMMA * np.max(q_sa)

                    loss_val, _ = session.run([action_net.loss, action_net.optimizer],
                                              feed_dict={action_net.input: inputs,
                                                         action_net.target: targets,
                                                         action_net.action: target_actions})
                    loss += loss_val
                    print_log_string("loss is %f, overall loss is %f\n" %
                                     (loss_val, loss / nb_iterations), 6, col=2)
                    print_log_string("total reward acquired is %f\n" % total_reward, 7, col=2)

                STDSRC.refresh()
                curses.echo()
                curses.nocbreak()

                if step % C == 0:
                    update_target_net(session)

                nb_iterations += 1
                sample = sample2
                step += 1

            avg_reward_list.append(total_reward/float(nb_iterations))
            total_reward_list.append(total_reward)
            ep_length.append(nb_iterations)

        saver.save(session, path + 'models/model-main.ckpt')
        with open('stats/avg_reward' + '.pkl', 'wb') as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)
        with open('stats/total_reward' + '.pkl', 'wb') as f:
            pickle.dump(total_reward_list, f, pickle.HIGHEST_PROTOCOL)
        with open('stats/ep_length' + '.pkl', 'wb') as f:
            pickle.dump(ep_length, f, pickle.HIGHEST_PROTOCOL)
    curses.endwin()

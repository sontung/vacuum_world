import tensorflow as tf
import pygame
import visulization
import controller
import skimage.io
import skimage.transform
import skimage.exposure
import skimage.color
import numpy as np
import model_tf


GAME = visulization.GUI()
CONTROLLER = controller.ManualController(GAME.get_robot())
ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "suck": 4
}
INDICES_TO_ACTIONS = {v: k for k, v in ACTIONS.iteritems()}


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


def test_mode():
    frame, _, _ = get_next_frame()
    pygame.display.set_mode((500, 500))
    GAME.display_surface = pygame.display.get_surface()
    pygame.display.update()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("action"):
            action_net = model_tf.deepQnet(lr=.1, nb_actions=5)
        with tf.variable_scope("target"):
            target_net = model_tf.deepQnet(lr=.1, nb_actions=5)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        print 'Loading Model...'
        saver.restore(sess, "models_ddqnmodels/model-main.ckpt")
        
        while True:
            sample = np.stack((frame, frame, frame, frame), axis=2)
            sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
            q_values = sess.run(target_net.output, feed_dict={target_net.input: sample})
            an_action = np.argmax(q_values)
            print "Predicted q values are", q_values
            action = INDICES_TO_ACTIONS[an_action]
            print "Executing action", action
            frame, _, _ = get_next_frame(action)
            pygame.display.update()
            pygame.time.wait(1000)


if __name__ == "__main__":
    test_mode()
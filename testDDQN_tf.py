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
    if action is not None:
        CONTROLLER.control(action)
    GAME.draw()
    img = pygame.surfarray.array3d(GAME.get_surface())
    skimage.io.imsave("frames/frame1.jpg", img)
    img = skimage.transform.resize(img, (84, 84, 3))
    s = (CONTROLLER.robot.get_tile_pos()[0], CONTROLLER.robot.get_tile_pos()[1], tuple(CONTROLLER.dirty_tiles))
    return img, CONTROLLER.reward_func(s), CONTROLLER.terminate()


def test_mode():
    frame, _, _ = get_next_frame()
    pygame.display.set_mode((500, 500))
    GAME.display_surface = pygame.display.get_surface()
    pygame.display.update()
    tf.reset_default_graph()
    mainQN = model_tf.Q_network()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        print 'Loading Model...'
        saver.restore(sess, "models_ddqn/model-main.ckpt")

        while True:
            sample = np.reshape(frame, [1, 84 * 84 * 3])
            q_values = sess.run(mainQN.Q_out, feed_dict={mainQN.scalarInput: sample})
            an_action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: sample})
            print "Predicted q values are", q_values
            action = INDICES_TO_ACTIONS[an_action[0]]
            print "Executing action", action
            frame, _, _ = get_next_frame(action)
            pygame.display.update()
            pygame.time.wait(1000)


if __name__ == "__main__":
    test_mode()
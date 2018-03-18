from keras.models import load_model
import pygame
import visulization
import controller
import skimage.io
import skimage.transform
import skimage.exposure
import skimage.color
import numpy as np

GAME = visulization.GUI()
CONTROLLER = controller.ManualController(GAME.get_robot())
MODEL = load_model('models/model1.h5')
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
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, (80, 80))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    s = (CONTROLLER.robot.get_tile_pos()[0], CONTROLLER.robot.get_tile_pos()[1], tuple(CONTROLLER.dirty_tiles))
    return img, CONTROLLER.reward_func(s), CONTROLLER.terminate()


def test_mode():
    frame, _, _ = get_next_frame()
    pygame.display.set_mode((500, 500))
    GAME.display_surface = pygame.display.get_surface()
    pygame.display.update()
    while True:
        img = pygame.surfarray.array3d(GAME.get_surface())
        skimage.io.imsave("frames/frame1.jpg", img)
        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img, (80, 80))
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
        sample = np.stack((img, img, img, img), axis=2)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
        q_values = MODEL.predict(sample)
        print "Predicted q values are", q_values

        action = INDICES_TO_ACTIONS[np.argmax(q_values)]
        print "Executing action", action
        CONTROLLER.key_handler(action)
        GAME.draw()
        pygame.display.update()
        # pygame.time.wait(1000)


if __name__ == "__main__":
    test_mode()

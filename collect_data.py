import pygame
import visulization
import controller
import skimage
import os


if __name__ == "__main__":
    gui = visulization.GUI()
    pygame.display.update()
    manual = controller.Controller(gui.get_robot())
    gui.draw()
    while True:
        gui.draw()
        manual.key_handler()
        s = (manual.robot.get_tile_pos()[0], manual.robot.get_tile_pos()[1], tuple(manual.dirty_tiles))
        print "reward is", manual.reward_func(s), manual.terminate()
        pygame.display.update()
        if len(manual.dirty_tiles) == 0:
            gui.draw()
            pygame.display.update()
            break
    print "Job done in %d moves" % manual.nb_moves

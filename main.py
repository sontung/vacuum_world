import pygame
import visulization
import controller


def value_iteration():
    gui = visulization.GUI()
    pygame.display.update()
    mdp = controller.MDPController(gui.get_robot(), gui.get_window_size())
    mdp.converge_utable()

    nb_moves = 0
    while True:
        if len(mdp.dirty_tiles) == 0:
            break
        nb_moves += 1
        gui.draw()
        mdp.decide_action("value iteration")
        pygame.display.update()
        pygame.time.wait(1000)
    print "Job done in %d moves" % nb_moves


def random():
    gui = visulization.GUI("problems/problem1.txt")
    pygame.display.update()
    random_controller = controller.RandomController(gui.get_robot())

    nb_moves = 0
    while True:
        nb_moves += 1
        gui.draw()
        random_controller.control()
        pygame.display.update()
        pygame.time.wait(10)
    print "Job done in %d moves" % nb_moves


def manual(action):
    gui = visulization.GUI()
    pygame.display.update()
    random_controller = controller.ManualController(gui.get_robot())

    gui.draw()
    pygame.display.update()
    pygame.time.wait(1000)
    random_controller.control(action)
    gui.draw()
    pygame.display.update()
    pygame.time.wait(1000)


def select_mode():
    gui = visulization.GUI("select")
    pygame.display.update()
    random_controller = controller.ManualController(gui.get_robot())
    while True:
        random_controller.key_handler()
        gui.draw()
        pygame.display.update()
        # pygame.time.wait(1000)


if __name__ == "__main__":
    # value_iteration()
    select_mode()

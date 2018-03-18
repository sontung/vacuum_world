import pygame
import random
import sys
import itertools
import pickle
import skimage.io
from pygame.locals import *


class Controller(object):
    def __init__(self, robot):
        self.movement = {
            K_UP: "up",
            K_DOWN: "down",
            K_RIGHT: "right",
            K_LEFT: "left"
        }
        self.actions = {
            "change_direction": self.change_direction,
            "move": self.move,
            "suck": self.suck
        }
        self.robot = robot
        self.dirty_tiles = []
        self.nb_moves = 0
        self.reward = 0.0
        self.prev_moves = 0
        for t in self.robot.get_tile_map():
            if self.robot.get_tile_map()[t].get_type() == "dirty":
                self.dirty_tiles.append((t[0], t[1]))
        self.nb_dirty_tiles = len(self.dirty_tiles)
        self.game = None

    def quit(self):
        pygame.quit()
        sys.exit()

    def key_handler(self):
        event = pygame.event.poll()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.quit()
            elif event.key == K_UP:
                self.change_direction("up")
                self.move()
            elif event.key == K_DOWN:
                self.change_direction("down")
                self.move()
            elif event.key == K_LEFT:
                self.change_direction("left")
                self.move()
            elif event.key == K_RIGHT:
                self.change_direction("right")
                self.move()
            elif event.key == K_s:
                self.suck()
            elif event.key == K_r:
                print "start recording"
                sys.stdout = open("data/%s/%s.txt" % (self.game, self.game), "w")
            elif event.key == K_t:
                sys.stdout = sys.__stdout__
                print "stop recording"
        elif event.type == MOUSEBUTTONDOWN:
            tile_pos = (event.pos[0] / 50, event.pos[1] / 50)
            self.dirty_tiles.append(tile_pos)
            self.nb_dirty_tiles = len(self.dirty_tiles)
            self.robot.get_tile_map()[tile_pos].tile_type = "dirty"

    def change_direction(self, new):
        self.robot.change_direction(new)

    def move(self, dis=50):
        self.nb_moves += 1
        for i in range(5):
            self.robot.move(dis / 5)

    def suck(self):
        s = (self.robot.get_tile_pos()[0], self.robot.get_tile_pos()[1], tuple(self.dirty_tiles))
        self.robot.suck()
        self.nb_moves += 1
        if (s[0], s[1]) in self.dirty_tiles:
            self.dirty_tiles.remove((s[0], s[1]))
            self.reward = 0.01

    def reward_func(self, s):
        """
        Return the reward for the current state of robot
        """
        reward = self.reward
        if reward > 0:
            self.reward = 0.0
        return reward

    def terminate(self):
        return len(self.dirty_tiles) == 0

    def save_screen(self):
        if self.nb_moves > self.prev_moves:
            img = pygame.surfarray.array3d(pygame.display.get_surface())
            skimage.io.imsave("data/%s/frame%d.jpg" % (self.game, self.nb_moves), img)
            self.prev_moves = self.nb_moves


class RandomController(Controller):
    def decide_action(self):
        prob = random.random()
        if prob <= 0.4:
            return "change_direction", random.choice(self.movement.values())
        else:
            return "suck", None

    def control(self):
        act, param = self.decide_action()
        if act == "change_direction":
            self.actions[act](param)
            self.actions["move"]()
        elif act == "suck":
            self.actions[act]()
        return act, param


class ManualController(Controller):
    def control(self, act):
        if act == "suck":
            self.suck()
        else:
            self.change_direction(act)
            self.move()


class MDPController(Controller):
    def __init__(self, robot, size):
        self.policy = None
        super(MDPController, self).__init__(robot)
        self.size = size[0]/50, size[1]/50

        print "Constructing U table ..."
        total = 0
        for i in range(len(self.dirty_tiles) + 1):
            total += len(list(itertools.combinations(self.dirty_tiles, i)))
        total = total*100
        states = []

        def state_generator(size1, size2, tiles):
            for x in range(size1):
                for y in range(size2):
                    for tile in tiles:
                        yield (x, y, tile)

        for i in range(len(self.dirty_tiles) + 1):
            dirty_tiles = list(itertools.combinations(self.dirty_tiles, i))
            generator = state_generator(self.size[0], self.size[1], dirty_tiles)
            while True:
                try:
                    states.append(generator.next())
                except StopIteration:
                    break
            print "%d/%d states generated" % (len(states), total)

        self.U_table = {s: 0 for s in states}

        # Temporal difference learning
        self.Q_value = {s: {a: 0 for a in self.get_possible_actions(s)} for s in states}
        self.N = {s: {a: 0 for a in self.get_possible_actions(s)} for s in states}

    def get_possible_actions(self, s):
        """
        Return list of applicable actions in a state
        :return:
        """
        actions = ["left", "right", "up", "down", "suck"]
        if (s[0], s[1]) not in s[2]:
            actions.remove("suck")
        if s[0] == 0:
            actions.remove("left")
        if s[1] == 0:
            actions.remove("up")
        if s[0] == self.size[0] - 1:
            actions.remove("right")
        if s[1] == self.size[1] - 1:
            actions.remove("down")
        return actions

    def transition_model(self, s, a):
        """
        Return state s' given s and a
        :param s: state
        :param a: action
        :return:
        """
        if a == "left":
            if (s[0]-1, s[1], tuple(self.dirty_tiles)) in self.U_table:
                return s[0]-1, s[1], s[2]
            else:
                return s
        elif a == "right":
            if (s[0]+1, s[1], tuple(self.dirty_tiles)) in self.U_table:
                return s[0]+1, s[1], s[2]
            else:
                return s
        elif a == "up":
            if (s[0], s[1]-1, tuple(self.dirty_tiles)) in self.U_table:
                return s[0], s[1]-1, s[2]
            else:
                return s
        elif a == "down":
            if (s[0], s[1]+1, tuple(self.dirty_tiles)) in self.U_table:
                return s[0], s[1]+1, s[2]
            else:
                return s
        elif a == "suck":
            if (s[0], s[1]) in s[2]:
                temp = list(s[2])
                temp.remove((s[0], s[1]))
                return s[0], s[1], tuple(temp)
            else:
                return s

    def converge_utable(self, gamma=0.2, error=0.0001):
        """
        Converge U table using value iteration algorithm
        :param gamma:
        :param error:
        :return:
        """
        print "Converging U table ..."
        nb_iter = 0
        while True:
            loss = 0
            nb_iter += 1
            for s in self.U_table:
                old_val = self.U_table[s]
                best_action = max(self.get_possible_actions(s),
                                  key=lambda x: self.U_table[self.transition_model(s, x)])
                self.U_table[s] = self.reward_func(s) + gamma*self.U_table[self.transition_model(s, best_action)]
                if abs(self.U_table[s] - old_val) > loss:
                    loss = abs(self.U_table[s] - old_val)
            print "Loss is %f" % loss
            if loss < error * (1 - gamma) / gamma:
                break
        print "Convergence done in %d iters with loss = %f" % (nb_iter, loss)
        return self.U_table

    def converge_policy(self, gamma=0.2):
        self.policy = {s: random.choice(self.get_possible_actions(s)) for s in self.U_table.keys()}
        nb_iter = 0
        print "Converging policy ..."
        while True:
            nb_changes = 0
            # policy evaluation
            for state in self.U_table:
                self.U_table[state] = self.reward_func(state) + \
                                      gamma*self.U_table[self.transition_model(state, self.policy[state])]

            # policy improvement
            for state in self.U_table:
                best_action = max(self.get_possible_actions(state),
                                  key=lambda x: self.U_table[self.transition_model(state, x)])
                if self.U_table[self.transition_model(state, best_action)] > \
                        self.U_table[self.transition_model(state, self.policy[state])]:
                    self.policy[state] = best_action
                    nb_changes += 1

            # terminate condition
            print "Loss is %d changes" % nb_changes
            if nb_changes == 0:
                break
            nb_iter += 1
        print "Convergence done in %d iterations" % nb_iter
        return self.policy

    def decide_action(self, algorithm):
        """
        Decide the best action based on the current state of robot
        :return:
        """
        s = (self.robot.get_tile_pos()[0], self.robot.get_tile_pos()[1], tuple(self.dirty_tiles))
        if algorithm == "value iteration":
            best_action = max(self.get_possible_actions(s),
                              key=lambda x: self.U_table[self.transition_model(s, x)])
        elif algorithm == "policy iteration":
            best_action = self.policy[s]
        elif algorithm == "q learning":
            best_action = self.q_learning()
        # print best_action, s

        if best_action == "right":
            self.robot.change_direction("right")
            self.robot.move()
        elif best_action == "left":
            self.robot.change_direction("left")
            self.robot.move()
        elif best_action == "up":
            self.robot.change_direction("up")
            self.robot.move()
        elif best_action == "down":
            self.robot.change_direction("down")
            self.robot.move()
        elif best_action == "suck":
            self.robot.suck()
            if (s[0], s[1]) in self.dirty_tiles:
                self.dirty_tiles.remove((s[0], s[1]))

    def converge_q_table(self, nb_iter=2000, load_old_data=False):
        if not load_old_data:
            nb_iter = 0
            while len(self.dirty_tiles) > 0:
                nb_iter += 1
                print nb_iter, len(self.dirty_tiles), len([self.N[x][y] for x in self.N for y in self.N[x]])

                self.decide_action("q learning")
            with open("Q_table" + '.pkl', 'wb') as f:
                pickle.dump(self.Q_value, f, pickle.HIGHEST_PROTOCOL)
            with open("N_table" + '.pkl', 'wb') as f:
                pickle.dump(self.N, f, pickle.HIGHEST_PROTOCOL)
            print "Computation saved!"
        else:
            with open("Q_table" + '.pkl', 'rb') as f:
                self.Q_value = pickle.load(f)
            with open("N_table" + '.pkl', 'rb') as f:
                self.N = pickle.load(f)
            print "Computation loaded!"

    def q_learning(self, alpha=0.01, gamma=0.1, Ne=5):
        current_s = (self.robot.get_tile_pos()[0], self.robot.get_tile_pos()[1], tuple(self.dirty_tiles))
        current_r = self.reward_func(current_s)
        if len(self.dirty_tiles) == 0:
            self.Q_value[current_s][None] = current_r
        if self.prev_s is not None:
            best_action = max(self.get_possible_actions(current_s),
                              key=lambda x: self.Q_value[current_s][x])
            self.N[self.prev_s][self.prev_a] += 1
            self.Q_value[self.prev_s][self.prev_a] += alpha * self.N[self.prev_s][self.prev_a] * \
                                                      (self.prev_r + gamma*self.Q_value[current_s][best_action] -
                                                       self.Q_value[self.prev_s][self.prev_a])
        possible_actions = [a for a in self.get_possible_actions(current_s) if self.N[current_s][a] < Ne]
        if len(possible_actions) == 0:
            next_action = best_action
        else:
            next_action = min(possible_actions, key=lambda x: self.N[current_s][x])
        self.prev_s, self.prev_a, self.prev_r = current_s, next_action, current_r
        return self.prev_a


if __name__ == "__main__":
    mdp = MDPController(None, (10, 10))
    print mdp.converge_utable()

import pygame
import random
import sys

COLORS = {"white": (255, 255, 255),
          "black": (41, 36, 33),
          "navy": (0, 0, 128),
          "red": (139, 0, 0),
          "blue": (0, 0, 255),
          "dark": (3, 54, 73),
          "yellow": (255, 255, 0),
          "turquoise blue": (0, 199, 140),
          "green": (0, 128, 0),
          "light green": (118, 238, 0),
          "turquoise": (0, 229, 238),
          "brown": (139, 69, 19),
          "gray": (211, 211, 211)}


class GUI:
    def __init__(self, dir_to_prev_states=None):
        pygame.init()
        self.tile_size = 50
        self.window_size = (self.tile_size * 10, self.tile_size * 10)
        self.display_surface = pygame.Surface(self.window_size)
        self.tiles = []
        self.pos_to_tiles = {}
        if dir_to_prev_states is None:
            self.init_tiles(0.5)
            self.save_state()
        elif dir_to_prev_states == "select":
            self.init_tiles(0.0)
        else:
            self.load_state(dir_to_prev_states)
        self.robot = Robot((0, 0), self.window_size, self.pos_to_tiles)

    def init_tiles(self, prob):
        for i in range(0, self.window_size[0], self.tile_size):
            for j in range(0, self.window_size[1], self.tile_size):
                a_tile = Tile(self.decide_dirty(prob), (i, j), self.tile_size)
                self.pos_to_tiles[(i / 50, j / 50)] = a_tile
                self.tiles.append(a_tile)

    def load_state(self, path):
        sys.stdin = open(path, "r")
        lines = sys.stdin.readlines()
        dirty_tiles = []
        for line in lines:
            tokens = line[:-1].split(" ")
            if tokens[0] != "size":
                dirty_tiles.append((int(tokens[0]), int(tokens[1])))
        for i in range(0, self.window_size[0], self.tile_size):
            for j in range(0, self.window_size[1], self.tile_size):
                if (i, j) in dirty_tiles:
                    a_tile = Tile("dirty", (i, j), self.tile_size)
                else:
                    a_tile = Tile("clean", (i, j), self.tile_size)
                self.pos_to_tiles[(i / 50, j / 50)] = a_tile
                self.tiles.append(a_tile)
        sys.stdin = sys.__stdin__

    def get_surface(self):
        return self.display_surface

    def save_state(self):
        """
        Save the sate of the current problem
        :return:
        """
        print "Saving states ..."
        sys.stdout = open("problems/problem1.txt", "w")
        print "size", self.window_size[0] / self.tile_size, self.window_size[1] / self.tile_size
        for t in self.tiles:
            if t.get_type() == "dirty":
                print t.get_pos()[0], t.get_pos()[1]
        sys.stdout = sys.__stdout__

    def decide_dirty(self, prob):
        """
        Decide if a tile is dirty with probability
        :param prob:
        :return:
        """
        if random.random() < prob:
            return "dirty"
        else:
            return "clean"

    def draw(self):
        self.display_surface.fill(COLORS["light green"])
        for tile in self.tiles:
            tile.draw(self.display_surface)
        self.display_surface.blit(self.robot.get_img(), self.robot.get_pos())

    def get_robot(self):
        return self.robot

    def get_window_size(self):
        return self.window_size


class Tile:
    def __init__(self, tile_type, pos, size):
        self.tile_type = tile_type
        self.pos = pos
        self.size = size
        self.dust = []

    def corners(self):
        """
        List of positions of 4 corners of a tile
        :return:
        """
        return [self.pos, (self.pos[0] + self.size, self.pos[1]),
                (self.pos[0] + self.size, self.pos[1] + self.size), (self.pos[0], self.pos[1] + self.size)]

    def get_pos(self):
        return self.pos

    def get_type(self):
        return self.tile_type

    def clean(self):
        self.tile_type = "clean"

    def draw(self, surface):
        """
        Draw the tile to the surface
        :param surface:
        :return:
        """
        if self.tile_type == "clean":
            pygame.draw.rect(surface, COLORS["gray"],
                             (self.pos[0], self.pos[1], self.size, self.size))
        elif self.tile_type == "dirty":
            pygame.draw.rect(surface, COLORS["brown"],
                             (self.pos[0], self.pos[1], self.size, self.size))
            if not self.dust:
                for i in range(5):
                    x = random.randint(self.pos[0] + 3, self.pos[0] + self.size - 3)
                    y = random.randint(self.pos[1] + 3, self.pos[1] + self.size - 3)
                    self.dust.append((x, y))
            for x, y in self.dust:
                surface.fill(COLORS["black"], ((x, y), (5, 5)))
        pygame.draw.lines(surface, COLORS["red"], True, self.corners(), 2)


class Robot:
    def __init__(self, pos, window_size, tile_map):
        self.pos = pos
        self.up_sprites = [pygame.image.load(s) for s in ["assets/up1.png", "assets/up2.png"]]
        self.down_sprites = [pygame.image.load(s) for s in ["assets/down1.png", "assets/down2.png"]]
        self.right_sprites = [pygame.image.load(s) for s in ["assets/side1.png", "assets/side2.png"]]
        self.left_sprites = [pygame.transform.flip(pygame.image.load(s), 1, 0)
                             for s in ["assets/side1.png", "assets/side2.png"]]
        self.direction = "right"
        self.dir_to_sprite = {
            "left": [self.left_sprites, (self.pos[0] + 5, self.pos[1] + 14), (-1, 0)],
            "right": [self.right_sprites, (self.pos[0] + 5, self.pos[1] + 14), (1, 0)],
            "up": [self.up_sprites, (self.pos[0] + 11, self.pos[1] + 9), (0, -1)],
            "down": [self.down_sprites, (self.pos[0] + 11, self.pos[1] + 9), (0, 1)]
        }
        self.current_img = 0
        self.window_size = window_size
        self.tile_map = tile_map

    def update_dict(self):
        self.dir_to_sprite = {
            "left": [self.left_sprites, (self.pos[0] + 5, self.pos[1] + 14), (-1, 0)],
            "right": [self.right_sprites, (self.pos[0] + 5, self.pos[1] + 14), (1, 0)],
            "up": [self.up_sprites, (self.pos[0] + 11, self.pos[1] + 9), (0, -1)],
            "down": [self.down_sprites, (self.pos[0] + 11, self.pos[1] + 9), (0, 1)]
        }

    def get_img(self):
        return self.dir_to_sprite[self.direction][0][self.current_img]

    def get_pos(self):
        self.update_dict()
        return self.dir_to_sprite[self.direction][1]

    def get_tile_pos(self):
        return self.pos[0] / 50, self.pos[1] / 50

    def get_tile_map(self):
        return self.tile_map

    def change_direction(self, d):
        """
        Change direction of robot
        :param d:
        :return:
        """
        self.direction = d

    def if_outside(self, pos):
        """
        Check if the pos is outside of the window, thus reject it if it is
        :param pos:
        :return:
        """
        if pos[0] > self.window_size[0] - 50 or pos[0] < 0:
            return True
        elif pos[1] > self.window_size[1] - 50 or pos[1] < 0:
            return True
        else:
            return False

    def move(self, distance=50):
        """
        Move the robot to the next tile according to current direction
        """
        new_pos = tuple([self.pos[0] + distance * self.dir_to_sprite[self.direction][2][0],
                         self.pos[1] + distance * self.dir_to_sprite[self.direction][2][1]])
        if not self.if_outside(new_pos):
            self.pos = new_pos
            self.current_img = (self.current_img + 1) % 2

    def suck(self):
        self.tile_map[(self.pos[0] / 50, self.pos[1] / 50)].clean()


if __name__ == "__main__":
    gui = GUI()
    pygame.display.update()
    while True:
        gui.draw()
        pygame.display.update()
        pygame.time.wait(1000)
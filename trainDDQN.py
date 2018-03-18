from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from collections import deque

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
from keras import backend as K


for nb_ep in range(1000):
    INIT_EP = 1
    FINAL_EP = 0.1
    BATCH_SIZE = 32
    LR = 0.0001
    NB_OBSERVATIONS = 3200
    NB_FRAMES_TO_STOP_EXPLORING = 10000
    GAMMA = 0.8
    REPLAY_MEMORY_LIMIT = 50000
    UPDATE_FREQ = 4
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


    def build_model(weights=None):
        print "Building model ..."
        model = Sequential()
        model.add(Convolution2D(32, 8, strides=4, padding='same', input_shape=(80, 80, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(len(ACTIONS)))

        if weights is not None:
            model.load_weights(weights)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        print "Model built!"
        print model.summary()
        return model


    def copy_weights(from_model, to_model):
        from_model.save_weights('models/copy.h5')
        to_model.load_weights("models/copy.h5")

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


    def train(model, target_model):
        replay_memory = deque()

        # Get frame
        img, reward, terminate = get_next_frame()

        sample = np.stack((img, img, img, img), axis=2)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
        epsilon = INIT_EP

        nb_iterations = 0
        while not CONTROLLER.terminate():
            print "Doing iteration %d-th" % nb_iterations
            loss = 0

            # Choose action based on epsilon
            action = np.zeros(len(ACTIONS))
            if random.random() <= epsilon:
                print "Exploring ... with epsilon = %f" % epsilon
                index = random.choice(ACTIONS.values())
                action[index] = 1.0
            else:
                print "Exploiting ... with epsilon = %f" % epsilon
                q_values = model.predict(sample)
                print q_values
                index = np.argmax(q_values)
                action[index] = 1.0

            # Decrease epsilon
            if epsilon > FINAL_EP and nb_iterations > NB_OBSERVATIONS:
                epsilon -= (INIT_EP - FINAL_EP) / NB_FRAMES_TO_STOP_EXPLORING

            # Execute action and get next frame
            action_in_string = INDICES_TO_ACTIONS[index]
            print "  executing action %s" % action_in_string
            img2, reward2, terminate2 = get_next_frame(action_in_string)
            img2 = img2.reshape(1, img2.shape[0], img2.shape[1], 1)  # 1x80x80x1
            sample2 = np.append(img2, sample[:, :, :, :3], axis=3)
            print "  get reward %f" % reward2

            # Store transition in replay memory
            replay_memory.append((sample, index, reward2, sample2, terminate2))
            print "  replay memory has %d samples" % len(replay_memory)
            if len(replay_memory) > REPLAY_MEMORY_LIMIT:
                replay_memory.popleft()

            # Train when observe enough
            if nb_iterations > NB_OBSERVATIONS:
                mini_batch = random.sample(replay_memory, BATCH_SIZE)
                inputs = np.zeros((BATCH_SIZE, 80, 80, 4))
                targets = np.zeros((BATCH_SIZE, len(ACTIONS)))
                print "  Training on", inputs.shape, targets.shape
                for i in range(0, len(mini_batch)):
                    state_t = mini_batch[i][0]
                    action_t = mini_batch[i][1]  # This is action index
                    reward_t = mini_batch[i][2]
                    state_t1 = mini_batch[i][3]
                    terminal = mini_batch[i][4]

                    inputs[i] = state_t

                    targets[i] = model.predict(state_t)  # Hitting each buttom probability
                    q_sa = model.predict(state_t1)

                    targets[i, action_t] = reward_t + int(terminal)*GAMMA*target_model.predict(state_t1)[np.argmax(q_sa)]

                loss += model.train_on_batch(inputs, targets)
                print "  Loss is %f" % loss

                # Save model every 1000th iteration
                if nb_iterations % 1000 == 0:
                    print "  Saving model ..."
                    model.save("models/model1.h5")

                # Copy weights from q net to target net
                if nb_iterations % UPDATE_FREQ == 0:
                    copy_weights(model, target_model)

            nb_iterations += 1
            sample = sample2

        print "  Saving final model ..."
        model.save("models/model1.h5")
        sys.stdout = open("models/log.txt", "w")
        print "Episode %d is done in %d iterations with loss=%f" % (nb_ep, nb_iterations, loss)
        sys.stdout = sys.__stdout__


    if nb_ep == 0:
        q_net = build_model(weights=None)
        target_net = build_model(weights=None)
        train(q_net, target_net)
    else:
        q_net = build_model(weights="models/model1.h5")
        target_net = build_model(weights=None)
        copy_weights(q_net, target_net)
        train(q_net, target_net)
    K.clear_session()

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import os
import numpy as np
import cv2
import PIL
import keras
import random
import queue
from keras import optimizers
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Activation


GAMMA = 0.99
OBSERVATION = 50000.
EXPLORE = 100000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
ACTIONS=2
LEARNING_RATE=1e-4

class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path='/usr/share/man/man1/google-chrome-stable.1.gz', chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.set_window_size(200, 300)
        game_url='//chromedino.com/'
        self._driver.get(os.path.abspath(game_url))
        if custom_config:
            self._driver.execute_script("Runner.config.ACCELERATION=0")

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

        time.sleep(0.25)


    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(
            score_array)
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()

class DinoAgent:
    def __init__(self,game):
        self._game = game
        self.jump()
        time.sleep(.5)
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()

class Game_sate:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game

    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.1 * score / 10
        is_over = False
        if actions[1] == 1:
            self._agent.jump()
            reward = 0.1 * score / 11
        image = grab_screen()

        if self._agent.is_crashed():
            self._game.restart()
            reward = -11 / score
            is_over = True
        return image, reward, is_over


def grab_screen(_driver=None):
    screen = np.array(PIL.ImageGrab.grab(bbox=(40, 180, 440, 400)))
    image = process_img(screen)
    return image


def process_img(image):

    image = cv2.resize(image,(20,40))
    image = image[2:38, 10:50]
    image = cv2.Canny(image, threshold1=100, threshold2=200)
    return image

def buildmodel():
    print("Now we build the model")
    model = keras.Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(20,40,4)))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model




def trainNetwork(model, game_state):
    D = queue.deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).reshape(1, 20, 40,
                                                         4)

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    t = 0
    while (True):

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = model.predict(s_t)
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[action_index] = 1
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, r_t, terminal = game_state.get_state(a_t)
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()


        if t > OBSERVE:
            trainBatch(random.sample(D, BATCH))
        s_t = s_t1
        t = t + 1
        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa),
              "/ Loss ", loss)

def trainBatch(minibatch,model):
    global s_t
    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
    targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2
    loss = 0

    for i in range(0, len(minibatch)):
        state_t = minibatch[i][0]
        action_t = minibatch[i][1]
        reward_t = minibatch[i][2]
        state_t1 = minibatch[i][3]
        terminal = minibatch[i][4]
        inputs[i:i + 1] = state_t
        targets[i] = model.predict(state_t)
        Q_sa = model.predict(state_t1)
        if terminal:
            targets[i, action_t] = reward_t
        else:
            targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

    loss += model.train_on_batch(inputs, targets)

def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino,game)
    model = buildmodel()
    trainNetwork(model,game_state)

playGame(observe=False)
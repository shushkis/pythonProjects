import cv2
import time
import random
import threading
import numpy as np


from pygame.locals import *
import pygame, sys, math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DONE = False
TIMES_TO_PLAY = 10
STOP_SLEEP = False

def quiet_thread():
    global DONE
    while True:
        if DONE:
            break
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                DONE = True

def sleep_thread(seconds_to_sleep=5):
    for i in range(seconds_to_sleep):  # using loop to terminate
        global STOP_SLEEP
        if STOP_SLEEP:
            break
        time.sleep(1)




class AttentionGame:
    def __init__(self):
        self.easy_stats = []
        self.mid_stats = []
        self.hard_stats = []


        self.radius = 50

        center1 = (100, 150)
        center2 = (400, 150)

        self.centers = [center1, center2]

        self.circle_thickness = -1  # Line thickness of -1 fill whole circle

        self.is_inside = False
        self.clicked   = False

        self.black = 0, 0, 0
        self.white = 255, 255, 255
        red = 255, 0, 0
        green = 0, 255, 0
        blue = 0, 0, 255

        self.colors = {"blue": blue, "green": green, "red": red}
        self.chosen_color = None



        # Game window
        pygame.init()
        self.screen = pygame.display.set_mode((512, 255))
        pygame.display.set_caption('Attention Game')
        # font -- text
        pygame.font.init()
        self.font = pygame.font.Font('freesansbold.ttf', 32)  # font and size

        pygame.display.update()
        self.screen.fill(self.white)


    def is_inside_circle(self, circle_center, radius):
        # (x - center_x)² + (y - center_y)² < radius²
        self.is_inside = False
        x = pygame.mouse.get_pos()[0]
        y = pygame.mouse.get_pos()[1]
        self.is_inside = (((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2) < radius ** 2)

    def draw_circles(self, color, center):
        pygame.draw.circle(self.screen, color, center, self.radius)
        pygame.display.update()

    def easy_task(self):
        self.chosen_color = random.choice(list(self.colors.keys()))
        message = "Click on the {} circle".format(self.chosen_color)
        textsurface = self.font.render(message, False, self.colors[self.chosen_color])
        self.screen.blit(textsurface, (0, 0))
        stats = self.task()
        self.easy_stats.append(stats)

    def task (self):
        global STOP_SLEEP
        stats = 0
        for color in random.sample(list(self.colors.values()), k=2):
            if color == self.colors[self.chosen_color]:
                continue
            first_ceter = random.choice(self.centers)
            second_center = self.centers[self.centers.index(first_ceter) - 1]

            self.draw_circles(self.colors[self.chosen_color], first_ceter)
            self.draw_circles(color, second_center)

            start_time = time.time()

            STOP_SLEEP = False
            sleep_thr = threading.Thread(target=sleep_thread)
            sleep_thr.start()

            while sleep_thr.is_alive():  # block here until user clicked or timer ended
                if pygame.mouse.get_pressed()[0]:
                    self.is_inside_circle(first_ceter, self.radius)
                    STOP_SLEEP = True
                    break
            if self.is_inside:
                stats = time.time() - start_time
            else:
                stats = -1
            sleep_thr.join()  #wait to finish
            return stats

    def hard_task(self):
        self.chosen_color = random.choice(list(self.colors.keys()))
        second_color       = random.choice(list(self.colors.keys()))  # choose other so text will say choosen_color
        message = "Click on the "
        textsurface1 = self.font.render(message, True, self.black)
        self.screen.blit(textsurface1, (0, 0))

        textsurface2 = self.font.render(self.chosen_color, True, self.colors[second_color])
        self.screen.blit(textsurface2, (200, 0))

        message = " circle"
        textsurface3 = self.font.render(message, True, self.black)
        self.screen.blit(textsurface3, (285, 0))

        stats = self.task()
        self.hard_stats.append(stats)

    def print_stat(self):
        plt.plot([x for x in range (len(self.easy_stats))], self.easy_stats , label="easy")
        plt.plot([x for x in range (len(self.hard_stats))], self.hard_stats,label="hard")
        plt.title('Attention statistics')
        plt.legend()
        plt.show()

    def stage_counter(self, text):
        for i in range(10):
            self.screen.fill(self.white)  # clear screen
            message = "Starting {} in".format(text)
            textsurface = self.font.render(message, True, self.black)
            self.screen.blit(textsurface, (0, 0))

            textsurface = self.font.render(str(10-i), False, self.black)
            self.screen.blit(textsurface, (250, 0))
            pygame.display.update()
            time.sleep(1)

    def run(self):
        global DONE

        # quiet thread
        q_event = threading.Thread(target=quiet_thread, args=())
        q_event.start()

        message1 = "You will need to click "
        message2 = "the right color circle"
        textsurface1 = self.font.render(message1, True, self.black)
        textsurface2 = self.font.render(message2, True, self.black)
        self.screen.blit(textsurface1, (0, 0))
        self.screen.blit(textsurface2, (0, 50))
        pygame.display.update()
        time.sleep(5)

        self.stage_counter("easy")

        for i in range(TIMES_TO_PLAY):  # easy task mode
            self.screen.fill(self.white)  # clear screen
            self.easy_task()
            if DONE:
                break

        self.stage_counter("hard")

        for i in range(TIMES_TO_PLAY):  # hard task mode
            self.screen.fill(self.white)  # clear screen
            self.hard_task()
            if DONE:
                break

        DONE = True # to term all running threads
        self.print_stat()
        pygame.quit()
        return



if __name__ == '__main__':
    game_obj = AttentionGame()
    game_obj.run()
    sys.exit()
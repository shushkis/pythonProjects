import sys

import time
import numpy as np
import pygame
from matplotlib import pyplot as plt

import random

from tkinter import *
from tkinter import messagebox

# globals
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
MAGENTA = ((255,0,230))
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')


WINDOW_HEIGHT = 1000
WINDOW_WIDTH = 1000
BOARD = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

GRID_SIZE = 200  # Set the size of the grid block


PROPABILITY_TO_GET_SICK = 0.01
PROPABILITY_TO_GET_SICK_HIGH = 0.3
PROPABILITY_TO_GET_SICK_LOW = 0.01
OUTBRAKE_THREASHOLD = 0.2

PROPABILITY_TO_START_AS_SICK = 0.01
PROPABILITY_TO_WALK_FAST = 0.1
MAX_AGE_TO_BE_CONTAGIOUS = 14
NUM_OF_CREATURES = 0.5

block_size = 10

background_color = (30, 30, 30)
PAUSE_SIM = False

ITERATIONS = 10000



STOP_THREADS = False

# ---------------------------------- GUI --------------------------------------------
class Screen:
    def __init__(self, title, width=600, height=600, fill=BLACK):
        self.title = title
        self.width = width
        self.height = height
        self.fill = fill
        self.current = False

    def make_current(self):
        pygame.display.set_caption(self.title)
        self.current = True
        self.screen = pygame.display.set_mode((self.width, self.height))

    def end_current(self):
        self.current = False

    def is_active(self):
        return self.current

    def update(self,fill):
        fill = fill if fill != None else self.fill
        if (self.current):
            self.screen.fill(fill)

    def blit(self,*args):
        self.screen.blit(*args)


class InputBox:
    # https://stackoverflow.com/questions/46390231/how-can-i-create-a-text-input-box-with-pygame
    def __init__(self, x, y, w, h, text='', can_be_edited=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = FONT.render(text, True, self.color)
        self.active = False
        self.can_be_edited = can_be_edited
        # Cursor declare
        self.txt_rect = self.txt_surface.get_rect()
        self.cursor = pygame.Rect(self.txt_rect.topright, (3, self.txt_rect.height + 2))

    def handle_event(self, event):
        if not self.can_be_edited:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(150, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen.screen, self.color, self.rect, 2)
        # Blit the  cursor
        if self.active and self.can_be_edited:
            if time.time() % 1 > 0.5:
                # bounding rectangle of the text
                text_rect = self.txt_surface.get_rect(topleft=(self.rect.x + 5, self.rect.y + 10))

                # set cursor position
                self.cursor.midleft = text_rect.midright

                pygame.draw.rect(screen.screen, self.color, self.cursor)

class Label:
    # https://nerdparadise.com/programming/pygame/part5
    def __init__(self, x, y, text="", font_size=20, color=MAGENTA):
        self.color = color
        self.text = text
        self.txt_surface = pygame.font.SysFont("comicsansms", font_size).render(text, True, self.color)
        self.active = False
        self.x = x
        self.y = y

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.x, self.y))

# ----------------------------------------------------------------------------------------------------

class Creature:
    global PROPABILITY_TO_START_AS_SICK, NUM_OF_CREATURES, MAX_AGE_TO_BE_CONTAGIOUS, OUTBRAKE_THREASHOLD, PROPABILITY_TO_GET_SICK_HIGH, PROPABILITY_TO_GET_SICK_LOW, PROPABILITY_TO_WALK_FAST
    def __init__(self, is_sick, can_walk_fast, pos):
        self.vals = [True,False]
        self.is_sick = is_sick
        self.is_contagious = 0 # counter that if equals to MAX_AGE zeros
        self.can_walk_fast = can_walk_fast
        self.was_sick = False
        self.pos = pos #(x,y)
        self.new_pos = ()
        self.moved = False
        self.age = 0


    def is_infected(self, probability, infected):
        coefficient = (1 - probability) ** infected
        num = random.random()
        if num > coefficient:
            return True
        return False

    def did_i_got_sick(self):
        return random.random() < PROPABILITY_TO_GET_SICK

    def update(self, has_sick_neighbor):
        self.age += 1
        if self.is_sick:
            if self.is_contagious >= MAX_AGE_TO_BE_CONTAGIOUS:
                self.is_sick = False
                self.was_sick = True
            else:
                self.is_contagious += 1
        else:
            if has_sick_neighbor and not self.was_sick:
                self.is_sick = self.did_i_got_sick()
        self.move()



    def move(self):
        # np.random.seed(1)
        faster = 10 if self.can_walk_fast else 0
        i = random.randint(-1,2) + faster
        j = random.randint(-1,2) + faster
        self.new_pos = ((self.pos[0] + i + GRID_SIZE) % GRID_SIZE, (self.pos[1] + j + GRID_SIZE) % GRID_SIZE)

    def draw(self):
        color = RED if self.is_sick else YELLOW if self.was_sick else GREEN
        rect = pygame.Rect(self.pos[0] * block_size, self.pos[1]* block_size, block_size, block_size)
        pygame.draw.rect(BOARD, color, rect,0,5)



class Grid:
    global PROPABILITY_TO_START_AS_SICK, NUM_OF_CREATURES, MAX_AGE_TO_BE_CONTAGIOUS, OUTBRAKE_THREASHOLD, PROPABILITY_TO_GET_SICK_HIGH, PROPABILITY_TO_GET_SICK_LOW, PROPABILITY_TO_WALK_FAST
    def __init__(self):
        self.rows = GRID_SIZE
        self.cols = GRID_SIZE
        self.num_of_sick = 0
        self.num_of_creatures = 0
        self.iteration_number = 0
        self.stats = {}
        self.flips = {}
        self.prev_flip = 0
        self.grid = self.init_grid()
        self.init_creatures_in_grid()


    def init_grid(self):
        """
        returns a grid of GRID_SIZExGRID_SIZE [True,False] values
        to init the creatures where True means that we will have a creature there
        """
        return np.random.choice([True,False], GRID_SIZE * GRID_SIZE, p=[NUM_OF_CREATURES, 1 - NUM_OF_CREATURES]).reshape(GRID_SIZE, GRID_SIZE)

    def init_creatures_in_grid(self):
        global OUTBRAKE_THREASHOLD
        self.stats[0] = 0
        self.num_of_creatures = np.count_nonzero(self.grid)
        sick_array = list(np.random.choice([True,False], np.count_nonzero(self.grid), p=[PROPABILITY_TO_START_AS_SICK, 1-PROPABILITY_TO_START_AS_SICK]))
        walk_faster_array = list(np.random.choice([True,False], np.count_nonzero(self.grid), p=[PROPABILITY_TO_WALK_FAST, 1-PROPABILITY_TO_WALK_FAST]))
        self.num_of_sick = len(sick_array)
        self.grid = self.grid.tolist()
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col]:  #if it's true we need to put a creature there
                    if len(sick_array) > 0:
                        is_sick = sick_array.pop()
                    else:
                        is_sick = False
                    if len(walk_faster_array) > 0:
                        can_walk_fast = walk_faster_array.pop()
                    else:
                        can_walk_fast = False
                    self.grid[row][col] = Creature(is_sick, can_walk_fast, (row,col))

        if self.num_of_sick / self.num_of_creatures > OUTBRAKE_THREASHOLD:
            self.prev_flip = -1
        else:
            self.prev_flip = 1
        self.flips[0] = self.prev_flip

    def has_sick_neighbor(self, position):
        x, y = position
        neighbour_cells = [(x - 1, y - 1), (x - 1, y + 0), (x - 1, y + 1),
                       (x + 0, y - 1), (x + 0, y + 1),
                       (x + 1, y - 1), (x + 1, y + 0), (x + 1, y + 1)]
        count = 0
        for x, y in neighbour_cells:
            if x >= 0 and y >= 0:
                try:
                    if self.grid[x][y] != 0:
                        count += self.grid[x][y].is_sick
                except:
                    pass

        return count > 0


    def update(self):
        self.stats[self.iteration_number] = self.num_of_sick
        self.num_of_sick = 0
        self.iteration_number += 1
        aux_grid = [0 for i in range(GRID_SIZE)]
        for i in range(GRID_SIZE):
            aux_grid[i] = [0 for i in range(GRID_SIZE)]
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col]:  #if it's true we have a creature to handle
                    creature =  self.grid[row][col]
                    creature.update(self.has_sick_neighbor(creature.pos))
                    aux_pos_x, aux_pos_y = creature.new_pos[0],creature.new_pos[1]
                    if aux_grid[aux_pos_x][aux_pos_y] == 0: #empty cell - First come first served.
                        creature.moved = True
                        aux_grid[aux_pos_x][aux_pos_y] = creature
                        # creature.pos = creature.new_pos
                        # creature.new_pos = ()
                    else: #conflict resolution
                        old_creature = aux_grid[aux_pos_x][aux_pos_y]
                        creatures = [old_creature, creature]
                        chosen_creature = random.choice(creatures)
                        chosen_creature.moved = True
                        aux_grid[aux_pos_x][aux_pos_y] = chosen_creature
                        creatures.remove(chosen_creature)
                        left_creature = creatures[0]
                        old_pos_x, old_pos_y= left_creature.pos[0], left_creature.pos[1] # stay in place, if you can
                        if aux_grid[old_pos_x][old_pos_y] == 0: #empty cell
                            left_creature.moved = True
                            left_creature.new_pos = left_creature.pos
                            aux_grid[old_pos_x][old_pos_y] = left_creature
                        else: #try to find an empty slot, starting from the corner.
                            new_pos_x, new_pos_y = self.get_random_new_pos(aux_grid, old_pos_x, old_pos_y,left_creature.can_walk_fast)
                            left_creature.moved = True
                            aux_grid[new_pos_x][new_pos_y] = left_creature

        self.grid = aux_grid

        #update creatures pos
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col]:
                    creature = self.grid[row][col]
                    assert (creature.moved) # if we are here creature must have moved
                    creature.moved = False
                    creature.pos = creature.new_pos
                    creature.new_pos = ()
                    self.num_of_sick += creature.is_sick

    def get_random_new_pos(self,grid, x, y, can_walk_fast):
        try_count = random.randint(1, 100)  # giving up to 100 tries to find new spot
        faster = 10 if can_walk_fast else 0
        while try_count > 0:
            i = random.randint(-1, 2) + faster
            j = random.randint(-1, 2) + faster
            row = (x + i  + try_count) % self.rows
            col = (y + j  + try_count) % self.cols
            if grid[row][col] == 0:
                return row, col
            try_count -= 1
        row = (x + 1 + self.rows) % self.rows
        col = (y + 1 + self.cols) % self.cols
        return row, col


    def draw(self):
        for x in range(0, WINDOW_WIDTH, block_size):
            row = int(x / block_size)
            for y in range(0, WINDOW_HEIGHT, block_size):
                col = int(y / block_size)
                if self.grid[row][col]:  # if it's true we have a creature there
                    rect = pygame.Rect(x, y, block_size, block_size)
                    pygame.draw.rect(BOARD, BLACK, rect, 1)
                    self.grid[row][col].draw()
                else:
                    rect = pygame.Rect(x, y, block_size, block_size)
                    pygame.draw.rect(BOARD, BLACK, rect,1)


    def end(self):
        global PROPABILITY_TO_START_AS_SICK, NUM_OF_CREATURES, MAX_AGE_TO_BE_CONTAGIOUS, OUTBRAKE_THREASHOLD, PROPABILITY_TO_GET_SICK_HIGH, PROPABILITY_TO_GET_SICK_LOW, PROPABILITY_TO_WALK_FAST
        # STOP_THREADS = False
        msg = "PROPABILITY_TO_START_AS_SICK = {}\n" \
              "PROPABILITY_TO_GET_SICK_LOW = {}\n " \
              "PROPABILITY_TO_GET_SICK_HIGH = {}\n"\
              "MAX_AGE_TO_BE_CONTAGIOUS = {}\n" \
              "OUTBRAKE_THREASHOLD = {}\n" \
              "NUM_OF_CREATURES = {}\n" \
              "PROPABILITY_TO_WALK_FAST={}".format(
                PROPABILITY_TO_START_AS_SICK,
                PROPABILITY_TO_GET_SICK_LOW,
                PROPABILITY_TO_GET_SICK_HIGH,
                MAX_AGE_TO_BE_CONTAGIOUS,
                OUTBRAKE_THREASHOLD,
                NUM_OF_CREATURES,
                PROPABILITY_TO_WALK_FAST)

        figs = []
        fig1 = plt.figure("Params")
        fig1.text(0.2, 0.4, msg)
        figs.append(fig1)

        fig2 = plt.figure("Graph of sick")
        ax2 = fig2.gca()
        ax2.plot(self.stats.keys(), [val / self.num_of_creatures for val in self.stats.values()])
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Num of sick")
        figs.append(fig2)

        fig3 = plt.figure("Graph of sick 2")
        ax3 = fig3.gca()
        ax3.plot(self.stats.keys(), self.stats.values())
        ax3.set_xlabel("Iterations")
        ax3.set_ylabel("Num of sick / total population")
        figs.append(fig3)

        fig4 = plt.figure("propabilty to get sick changed")
        ax4 = fig4.gca()
        ax4.plot(self.flips.keys(), self.flips.values())
        ax4.set_xlabel("Iterations")
        ax4.set_ylabel("flips ")
        figs.append(fig4)

        fig5 = plt.figure()
        fig5.canvas.mpl_connect('close_event', on_close)
        fig5.text(0.2, 0.5, 'Close Me to close all plots!', dict(size=10))
        figs.append(fig5)

        num_figs = len(figs)
        plt.show(block=False)
        plt.pause(1)
        while not STOP_THREADS:
            print("in while STOP_THREADS ", STOP_THREADS)
            if len(figs) == num_figs:
                plt.pause(1)
            if STOP_THREADS:
                for fig in figs:
                    plt.close(fig)
                figs = []
                plt.close('all')
                break

def on_close(event):
    '''
    to close all pyplots....
    :param event:
    :return:
    '''
    global STOP_THREADS
    STOP_THREADS = True
    print("STOP_THREADS ", STOP_THREADS)
    plt.close('all')

def validate_globals():
    '''
    validate user params so no stupid values entered.
    :return:
    '''
    global PROPABILITY_TO_START_AS_SICK, NUM_OF_CREATURES, MAX_AGE_TO_BE_CONTAGIOUS, OUTBRAKE_THREASHOLD, PROPABILITY_TO_GET_SICK_HIGH, PROPABILITY_TO_GET_SICK_LOW, PROPABILITY_TO_WALK_FAST
    if PROPABILITY_TO_START_AS_SICK >= 1:
        return False, "PROPABILITY_TO_START_AS_SICK must be smaller then 1"
    if NUM_OF_CREATURES >= 1: # I want it in density of the board
        NUM_OF_CREATURES = NUM_OF_CREATURES / (GRID_SIZE*GRID_SIZE)
    if NUM_OF_CREATURES > (GRID_SIZE*GRID_SIZE):
        return False, "NUM_OF_CREATURES must be in prectange (smaller then 1) or a number smaller then (GRID_SIZE*GRID_SIZE)"
    if PROPABILITY_TO_GET_SICK_HIGH >=1:
        return False, "PROPABILITY_TO_GET_SICK_HIGH must be smaller then 1"
    if PROPABILITY_TO_GET_SICK_LOW >=1:
        return False, "PROPABILITY_TO_GET_SICK_LOW must be smaller then 1"
    if PROPABILITY_TO_GET_SICK_LOW >= PROPABILITY_TO_GET_SICK_HIGH:
        return False, "PROPABILITY_TO_GET_SICK_LOW must be smaller then PROPABILITY_TO_GET_SICK_HIGH"
    if MAX_AGE_TO_BE_CONTAGIOUS < 1:
        return False, "MAX_AGE_TO_BE_CONTAGIOUS must be bigger then 1"
    if OUTBRAKE_THREASHOLD >= 1:
        return False, "OUTBRAKE_THREASHOLD must be smaller then 1"
    if PROPABILITY_TO_WALK_FAST >= 1:
        return False, "PROPABILITY_TO_WALK_FAST must be smaller then 1"
    return True,""

def intro(screen):
    '''
    Into screen to Corona Simulator
    :param screen:
    :return:
    '''
    global  PROPABILITY_TO_START_AS_SICK, NUM_OF_CREATURES, PROPABILITY_TO_GET_SICK_LOW, PROPABILITY_TO_GET_SICK_HIGH, MAX_AGE_TO_BE_CONTAGIOUS, OUTBRAKE_THREASHOLD, PROPABILITY_TO_WALK_FAST
    clock = pygame.time.Clock()
    label_1 = Label(20, 50, "PROPABILITY_TO_START_AS_SICK:")
    label_2 = Label(20, 100, "PROPABILITY_TO_GET_SICK_LOW:")
    label_3 = Label(20, 150, "PROPABILITY_TO_GET_SICK_HIGH:")
    label_4 = Label(20, 200, "MAX_AGE_TO_BE_CONTAGIOUS:")
    label_5 = Label(20, 250, "OUTBRAKE_THREASHOLD:")
    label_6 = Label(20, 300, "NUM_OF_CREATURES:")
    label_7 = Label(20, 350, "PROPABILITY_TO_WALK_FAST:")
    label_8 = Label(20, 400, "Green  = healthy", 15, GREEN)
    label_9 = Label(20, 420, "Yellow = was sick", 15, YELLOW)
    label_10 = Label(20, 440, "Red     = is sick", 15, RED)
    label_boxes = [label_1, label_2, label_3, label_4, label_5, label_6, label_7,label_8,label_9,label_10]

    input_box1 = InputBox(420, 50, 50, 32, str(PROPABILITY_TO_START_AS_SICK))
    input_box2 = InputBox(420, 100, 50, 32, str(PROPABILITY_TO_GET_SICK_LOW))
    input_box3 = InputBox(420, 150, 50, 32, str(PROPABILITY_TO_GET_SICK_HIGH))
    input_box4 = InputBox(420, 200, 50, 32, str(MAX_AGE_TO_BE_CONTAGIOUS))
    input_box5 = InputBox(420, 250, 50, 32, str(OUTBRAKE_THREASHOLD))
    input_box6 = InputBox(420, 300, 50, 32, str(NUM_OF_CREATURES))
    input_box7 = InputBox(420, 350, 50, 32, str(PROPABILITY_TO_WALK_FAST))
    btn_start  = InputBox(420, 450, 50, 32, "Start sim",can_be_edited=False)
    btn_quit   = InputBox(420, 550, 50, 32, "Quit")
    input_boxes = [input_box1, input_box2, input_box3, input_box4, input_box5, input_box6, input_box7, btn_start,btn_quit]
    done = False

    while not done:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for box in input_boxes:
                box.handle_event(event)


            if event.type == pygame.MOUSEBUTTONDOWN:
                # if the mouse is clicked on the button start the sim
                if 420 <= mouse[0] <=420 + 140 and 450 <= mouse[1] <= 450 + 40:
                    try:
                        PROPABILITY_TO_START_AS_SICK = float(input_box1.text)
                        PROPABILITY_TO_GET_SICK_LOW = float(input_box2.text)
                        PROPABILITY_TO_GET_SICK_HIGH = float(input_box3.text)
                        MAX_AGE_TO_BE_CONTAGIOUS = float(input_box4.text)
                        OUTBRAKE_THREASHOLD = float(input_box5.text)
                        NUM_OF_CREATURES = float(input_box6.text)
                        PROPABILITY_TO_WALK_FAST = float(input_box7.text)
                        is_valid, invalid_msg = validate_globals()
                        if is_valid:
                            done = True
                        else:
                            Tk().wm_withdraw()  # to hide the main window
                            messagebox.showerror('invalid value', 'invalid value {}'.format(invalid_msg))
                    except ValueError:
                        Tk().wm_withdraw()  # to hide the main window
                        messagebox.showerror("ValueError", "Looks like not all fields are integers")
                if 420 <= mouse[0] <= 420 + 140 and 550 <= mouse[1] <= 550 + 40:
                    pygame.quit()
                    sys.exit()

        for box in input_boxes:
            box.update()

        screen.update(background_color)
        for box in input_boxes:
            box.draw(screen)

        for box in label_boxes:
            box.draw(screen)

        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    global CLOCK
    pygame.init()
    pygame.font.init()
    FONT = pygame.font.SysFont('comicsansms', 20)

    menu_screen = Screen("Menu")
    sim_screen  = Screen("SIM", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    while True:
        win = menu_screen.make_current()
        CLOCK = pygame.time.Clock()

        intro(menu_screen)
        win = sim_screen.make_current()
        grid = Grid()
        grid.init_grid()
        grid.draw()

        sim_done = False
        while not sim_done:
            BOARD.fill(WHITE)
            grid.update()
            grid.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    grid.end()
                    sim_done = True
            if grid.num_of_sick == 0:
                print ("Exit because num_of_sick")
                grid.end()
                sim_done = True
            if grid.num_of_sick / grid.num_of_creatures > OUTBRAKE_THREASHOLD:
                if grid.prev_flip != -1:
                    grid.flips[grid.iteration_number] = -1
                    grid.prev_flip = -1
                PROPABILITY_TO_GET_SICK =  PROPABILITY_TO_GET_SICK_LOW
            else:
                if grid.prev_flip != 1:
                    grid.flips[grid.iteration_number] = 1
                    grid.prev_flip = 1
                PROPABILITY_TO_GET_SICK = PROPABILITY_TO_GET_SICK_HIGH

            pygame.display.update()



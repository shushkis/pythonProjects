import pygame
import random
import time, os,sys
from itertools import cycle
from geneticAlgo import GA
from matplotlib import pyplot as plt
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename

#https://github.com/jessicabp/FutoshikiSolver/tree/master/src

GIVEN_POSITIONS = {}
GIVEN_INEQUALITIES = {}

# colors
BLACK = (0, 0, 0)
DARKER_GREY = (50, 50, 50)
DARK_GREY = (100, 100, 100)
GREY = (150, 150, 150)
LIGHT_GREY = (200, 200, 200)
LIGHTER_GREY = (220, 220, 220)
WHITE = (255, 255, 255)

# pygame event.key:value mapping
keyMappings = {
    pygame.K_1: 1,
    pygame.K_2: 2,
    pygame.K_3: 3,
    pygame.K_4: 4,
    pygame.K_5: 5,
    pygame.K_6: 6,
    pygame.K_7: 7,
    pygame.K_8: 8,
    pygame.K_9: 9,
    pygame.K_BACKSPACE: ''
}

# pygame window

WINDOW_HEIGHT = 500
WINDOW_WIDTH = 500

blockSize = 100  # size of the grid cell
blockBorder = 25


class Rect:
    def __init__(self, rect, position, val=''):
        self.rect = rect
        self.val = val
        self.position = position
        self.horizontal = False
        self.textColor = BLACK

    def draw(self):
        pygame.draw.rect(SCREEN, self.color, self.rect)
        if self.val:
            self.drawVal(self.val)

    def drawVal(self, val):
        pygame.draw.rect(SCREEN, self.color, self.rect)  # draw over any previous val

        text = font.render(str(val), True, self.textColor)
        if self.horizontal:
            text = pygame.transform.rotate(text, -90)
        text_rect = text.get_rect(center=(self.rect.center))
        SCREEN.blit(text, text_rect)


class Cell(Rect):
    def __init__(self, rect, position):
        super().__init__(rect, position)
        self.color = LIGHT_GREY
        self.inequalities = {}
        self.isValid = None

    def setValidState(self, state):
        self.isValid = state
        return self

    # def validate(self, grid, visitedBefore=[]):
    def validateInequalities(self, grid):
        # """
        #     recursively evaluate adjacent cell values (if they have an inequality between them)
        #     will evaluate every linked cell along the chain of inequalities
        # """

        # visitedBefore.append(self) # so we dont backtrack during recursion
        validFlag = True

        for k, v in self.inequalities.items():
            # direction:ineq-object
            # cell-to-visit:how-to-evaluate-that-cell
            if k == 'up':
                posToVisit = (self.position[0] - 1, self.position[1])
                cellToVisit = grid.positions[posToVisit]

                leftCell, rightCell = cellToVisit, self
            elif k == 'down':
                posToVisit = (self.position[0] + 1, self.position[1])
                cellToVisit = grid.positions[posToVisit]

                leftCell, rightCell = self, cellToVisit

            elif k == 'left':
                posToVisit = (self.position[0], self.position[1] - 1)
                cellToVisit = grid.positions[posToVisit]

                leftCell, rightCell = cellToVisit, self

            elif k == 'right':
                posToVisit = (self.position[0], self.position[1] + 1)
                cellToVisit = grid.positions[posToVisit]

                leftCell, rightCell = self, cellToVisit

            if leftCell.val == '' or rightCell.val == '':
                continue  # cannot evaluate the inequality, skip

            # evaluate this cell's inequality
            if v.val == '':  # there is no inequality rect is empty, skip
                continue
            elif v.val == '<':
                evaluation = leftCell.val < rightCell.val
            elif v.val == '>':
                evaluation = leftCell.val > rightCell.val

            if evaluation == False:  # one invalid inequality affects the cell permanently
                validFlag = False
            leftCell.setValidState(validFlag)
            rightCell.setValidState(validFlag)

        return validFlag


class InequalitySign(Rect):
    def __init__(self, rect, position, neighbouringCells, horizontal):
        super().__init__(rect, position)
        self._states = cycle(['', '<', '>'])
        self.val = next(self._states)
        self.colorPalette = (DARK_GREY, LIGHTER_GREY)
        self.color = self.colorPalette[1]
        self.textColor = DARKER_GREY
        self.neighbouringCells = neighbouringCells
        self.horizontal = horizontal

    def cycleState(self):
        self.val = next(self._states)


class Grid:
    def __init__(self):
        self.height = int(WINDOW_HEIGHT / blockSize)
        self.width = int(WINDOW_WIDTH / blockSize)
        self.cells = []
        self.inequalities = []
        self.positions = {}  # mapping of position:cell/ineq pairs
        self.size = self.height
        self.stats = {}
        self.initial_numbers = GIVEN_POSITIONS
        self.solver = ''

    def start(self):
        self.createCells()
        self.createInequalities()
        self.linkCellsToInequalities()

    def draw(self):
        for cell in self.cells:
            cell.draw()

        for ineq in self.inequalities:
            ineq.draw()

    def num_constraints(self):
        sum_constraints = 0
        for inequalitie in self.inequalities:
            if inequalitie.val != '':
                sum_constraints += 1
        return sum_constraints


    def createCells(self):
        for iy, y in enumerate(range(0, WINDOW_HEIGHT, blockSize)):
            for ix, x in enumerate(range(0, WINDOW_WIDTH, blockSize)):
                position = (iy, ix)

                rect = pygame.Rect(x + blockBorder / 2, y + blockBorder / 2, blockSize - blockBorder, blockSize - blockBorder)

                cell = Cell(rect, position)
                if position in GIVEN_POSITIONS.keys():
                    cell.val = GIVEN_POSITIONS[position]
                self.cells.append(cell)
                self.positions[position] = cell

        return self.cells

    def createInequalities(self):
        # Vertical
        for i in range(self.width):  # in between horizontal spaces
            for j in range(self.height - 1):  # for every row
                neighbouringCells = ((i, j), (i, j + 1))
                position = (i, j + 0.5)
                x = blockSize * (j + 1) - blockBorder / 2
                y = blockSize * i + blockBorder / 2
                inequalityRect = pygame.Rect(x, y, blockBorder, blockSize - blockBorder)  # create rect

                inequality_sign = InequalitySign(inequalityRect, position, neighbouringCells, False)

                if position in GIVEN_INEQUALITIES.keys():
                    inequality_sign.val = GIVEN_INEQUALITIES[position]
                self.inequalities.append(inequality_sign)  # add inequalitysign to self.inequalities
                self.positions[position] = inequality_sign

        # Horizontal
        for i in range(self.width - 1):  # in between vertical spaces
            for j in range(self.height):  # for every column
                neighbouringCells = ((i, j), (i + 1, j))
                position = (i + 0.5, j)
                x = blockSize * j + blockBorder / 2
                y = blockSize * (i + 1) - blockBorder / 2


                inequalityRect = pygame.Rect(x, y, blockSize - blockBorder, blockBorder)  # create rect
                inequality_sign = InequalitySign(inequalityRect, position, neighbouringCells,True)
                if position in GIVEN_INEQUALITIES.keys():
                    inequality_sign.val = GIVEN_INEQUALITIES[position]
                self.inequalities.append(inequality_sign)  # add inequalitysign to self.inequalities

        return self.inequalities

    def linkCellsToInequalities(self):
        for index, cell in enumerate(self.cells):
            # find ineq with pos in ineq.neighbouringCells
            linkableIneq = [ineq for ineq in self.inequalities if cell.position in ineq.neighbouringCells]
            # identify their pos relative to this cell
            # ineqLocations = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
            ineqLocations = {}
            for ineq in linkableIneq:
                direction = self.findRelativePosition(ineq.position, cell.position)
                ineqLocations[direction] = ineq
            # tag them to Cell.inequalities
            self.cells[index].inequalities = ineqLocations

    @staticmethod
    def findRelativePosition(pos1, pos2):
        """
            takes 2 coordinates which are adjacent to each other
            returns position of pos1 relative to pos2: up, down, left, right
            larger values in coords means closer to bottom right
            [ (0,0) (0, 1)]
            [ (1,0) (1, 1)]
        """
        if pos1[0] < pos2[0]:  # pos1 is above pos2
            return 'up'
        if pos1[0] > pos2[0]:  # pos1 is below pos2
            return 'down'
        if pos1[1] < pos2[1]:  # pos1 is to the left of pos2
            return 'left'
        if pos1[1] > pos2[1]:  # pos1 is to the right of pos2
            return 'right'

    def validateCell(self, cell):
        if cell.val == '':
            validFlag = None
        else:
            validFlag = True

            posList = []
            for i in range(self.width):  # check row
                posList.append((cell.position[0], i))
            for i in range(self.height):  # check col
                posList.append((i, cell.position[1]))

            for posToCheck in posList:
                cellToCheck = self.positions[posToCheck]
                if cellToCheck != cell:
                    isRepeatValue = cellToCheck.val == cell.val
                    if isRepeatValue:
                        validFlag = False

            # check inequalities
            ineqResult = cell.validateInequalities(self)
            if not ineqResult:
                validFlag = False

        # cell.setValidState(validFlag)

        return validFlag

    def randomSetup(self):
        for ineq in self.inequalities:
            val = random.choices(
                population=['', '<', '>'],
                weights=[0.9, 0.05, 0.05]
            )[0]  # returns a 1 element list

            while ineq.val != val:
                ineq.cycleState()

        for cell in self.cells:
            val = random.choices(
                population=['',1,2,3,4,5],
                weights=[0.9] + 5*[0.05]
            )[0]  # returns a 1 element list

            cell.val = val

    def validateGrid(self):
        validFlag = True
        cellsValidity = []

        for index, cell in enumerate(self.cells):
            result = self.validateCell(cell, setColor=False)

            cellEvaluation = result and cell.val != ''

            if cellEvaluation == False:  # one invalid cell will invalidate the whole grid permanently
                validFlag = False

            cellsValidity.append(cellEvaluation)
            print(f"Evaluation for Cell @ {cell.position}: {cellEvaluation}")

        return validFlag

    def genetic_solve(self, fitness_func):
        ga = GA(self)
        self.solver = fitness_func
        return ga.solve(fitness_func)

    def recursion_solve(self, i, j):
        self.solver = 'recursive'
        # manage position of currently focused cell
        while self.positions[(i, j)].val != '':
            if j < self.width - 1:  # move right
                j += 1
            elif j == self.width - 1 and i < self.height - 1:  # move to next row
                j = 0
                i += 1
            elif j == self.width - 1 and i == self.height - 1:  # end of grid
                return True

        cell = self.positions[(i, j)]

        pygame.event.pump()

        for value in range(1, self.width + 1):  # check all possible values
            cell.val = value

            if self.validateCell(cell) == True:  # use first valid value found
                print(f"settled on {value} for {(i, j)}")
                cell.setValidState(True)

                self.draw()  # redraw grid

                pygame.display.update()
                pygame.time.delay(20)

                if self.recursion_solve(i, j) == True:  # recursion: will move to next empty cell
                    return True
                # else: # recursion returned false, backtrack

                self.draw()  # redraw grid

                pygame.display.update()
                pygame.time.delay(50)

        cell.val = ''
        cell.setValidState(None)

        return False  # no solution found, return false


def get_user_board(input_lines):
    global WINDOW_WIDTH, WINDOW_HEIGHT, GIVEN_POSITIONS
    num_lines = len(input_lines)
    board_size = int(input_lines[0])
    if board_size not  in [5,6,7]: # was told that board can 5x5,6x6 or 7x7
        messagebox.showerror("Error", "board_size must be 5x5,6x6 or 7x7")
        sys.exit()
    WINDOW_WIDTH = WINDOW_HEIGHT = board_size * blockSize
    num_given_digits = int(input_lines[1])
    if num_given_digits > 0:
        for i in range(2, 2+num_given_digits):
            given_digits = (list(map(int,input_lines[i].rsplit())))
            # given_digits = [x-1 for x in given_digits]
            GIVEN_POSITIONS[(given_digits[0] -1, given_digits[1]-1)] = given_digits[2] # in example file start from index 1?!
    num_given_grater_sings = int(input_lines[2 + num_given_digits])
    if num_given_grater_sings > 0:
        for i in range(3+num_given_digits, num_lines):
            given_grater_sign = (list(map(int,input_lines[i].rsplit())))
            given_grater_sign =  [x-1 for x in given_grater_sign] # in example file start from index 1?!
            if given_grater_sign[0:2] < given_grater_sign[2:4]:
                new_pos = tuple([sum(x)/2 for x in zip (given_grater_sign[0:2],given_grater_sign[2:4])])
                GIVEN_INEQUALITIES[new_pos] = '>'
            else:
                new_pos = tuple([sum(x) / 2 for x in zip(given_grater_sign[2:4], given_grater_sign[0:2])])
                GIVEN_INEQUALITIES[new_pos] = '<'



def main():
    global SCREEN, CLOCK, WINDOW_WIDTH, WINDOW_HEIGHT
while True:
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    ans = messagebox.askokcancel("Welcome", "Game keys are: \n\nF1 – regular genetic algorithm\nF2 – Darwin\nF3 – Lamarck\nF4 – recursive\nTab – for random board\nr – reset same board\n "
                                      "\nclick OK to choose file or Cancel to exit")
    if not ans:
        sys.exit()
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    if os.path.exists(filename):
        with open(filename,'r') as f:
            lines = f.readlines()
    get_user_board(lines)

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 60)

    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(WHITE)

    grid = Grid()
    grid.start()

    print_stat = False
    ended = False

    while not ended:
        grid.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ended = True
                pygame.quit()


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    grid.randomSetup()

                if event.key == pygame.K_r:
                    grid.start()

                if event.key == pygame.K_F1:
                    print_stat = True
                    t0 = time.time()
                    solution = grid.genetic_solve('regular')
                    t1 = time.time()
                    duration = round(t1 - t0, 2)

                if event.key == pygame.K_F2:
                    print_stat = True
                    t0 = time.time()
                    solution = grid.genetic_solve('darwin')
                    t1 = time.time()
                    duration = round(t1 - t0, 2)

                if event.key == pygame.K_F3:
                    print_stat = True
                    t0 = time.time()
                    solution = grid.genetic_solve('lamarck')
                    t1 = time.time()
                    duration = round(t1 - t0, 2)

                if event.key == pygame.K_F4:
                    print_stat = True
                    t0 = time.time()
                    solution = grid.recursion_solve(0, 0)
                    t1 = time.time()
                    duration = round(t1 - t0, 2)



        if print_stat:
            if solution:
                title = (f'Solved in {duration} seconds')
            else:
                title = (f'Confirmed impossible in {duration} seconds')
            plt.plot(grid.stats.keys(), grid.stats.values())
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title(grid.solver + " genetic algo -- " + title)
            plt.show()
            print_stat = False
        if not ended:
            pygame.display.update()


main()
import numpy as np
from random import  randint, choices
import pygame


# Constants for GA

POPULATION = 100

CROSSOVER_PROB_HIGH = 80
MUTATION_PROB_HIGH  = 50

CROSSOVER_PROB_LOW = 20
MUTATION_PROB_LOW  = 20

CROSSOVER_PROB = CROSSOVER_PROB_LOW
MUTATION_PROB  = MUTATION_PROB_LOW


MAX_LIFETIME = 10000

class GA():
    def __init__(self, grid):
        self.sum_constraints = 10 + 25 #grid.num_constraints()#self.max_value()
        self.grid = grid
        self.gen = []
        self.best_fitness = -np.inf
        self.worst_fitness = -1
        self.sum_fitness = 0
        self.too_many_with_same_score = 0
        self.index_best_sol = -1
        self.bests_solution = {}
        self.fitness_score = {}
        self.size = grid.size
        self.given_numbers = self.grid.initial_numbers
        self.initialize_generation()

    def initialize_generation(self):
        for i in range(POPULATION):
            self.gen.append(self.random_solution())


    def solve(self, fitness_func):
        global CROSSOVER_PROB, MUTATION_PROB
        generation = 0
        stoll_value = 0
        ever_refreshed = False
        while self.best_fitness <= self.sum_constraints:
            self.grid.stats[generation] = self.best_fitness
            if self.best_fitness == self.sum_constraints: # to save last value
                return True
            generation += 1
            self.fitness_score = self.choose_fitness(fitness_func)
            self.grid.draw()
            pygame.display.update()
            pygame.time.delay(1)
            pygame.event.pump()

            self.gen = self.new_generation()
            if self.best_fitness not in self.bests_solution.keys():
                self.bests_solution[self.best_fitness] = 0
            self.bests_solution[self.best_fitness] += 1

            # need_to_refresh = (self.fitness_score.count(self.best_fitness) > self.sum_constraints)
            # need_to_refresh = (list(self.grid.stats.values()).count(self.best_fitness) > generation / self.size) #POPULATION * stoll_value)
            # need_to_refresh = self.best_fitness > 0 and self.bests_solution.count(self.best_fitness) > POPULATION / 2
            # need_to_refresh = generation > 100 and ((generation %1000 == 0) or self.too_many_with_same_score > POPULATION / 5)
            need_to_refresh_hard = self.bests_solution[self.best_fitness] > POPULATION / 2  #need_to_refresh and (self.too_many_with_same_score > POPULATION or len(self.fitness_score.keys()) < 5)
            if need_to_refresh_hard == True:
                # ever_refreshed = True
                print ("!!! refresh gen !!!")
                self.too_many_with_same_score = 0
                self.best_fitness = -np.inf
                self.worst_fitness = -1
                self.sum_fitness = 0
                self.index_best_sol = -1
                self.bests_solution = {}
                self.fitness_score = {}
                best_sol = self.gen[self.index_best_sol]
                worst_sol = self.gen[self.index_worst_sol]
                self.gen = []
                self.gen.append(best_sol)
                self.gen.append(worst_sol)
                self.gen.append(best_sol.T)
                self.gen.append(worst_sol.T)
                self.initialize_generation()

            # if need_to_refresh:
            #     stoll_value += 1
                # # push best solution to pool -- selection
                # best_sol = self.gen[self.index_max_sol]
            #     # for i in range (5):
            #     #     random_index = randint(0, len(self.gen)-1)
            #     #     self.gen[random_index] = best_sol
                #remove best solution
                # self.gen[self.index_max_sol] = self.random_solution()
            #     # for i in range (5):
            #     #     random_index = randint(0, len(self.gen)-1)
            #     #     self.gen[random_index] = self.random_solution()


            # toggling crossover and mutation probabilities to avoid local maximum
            if (generation > 10 and (generation % 100 < 10)) or (not need_to_refresh_hard and self.bests_solution[self.best_fitness] > POPULATION / 4):
                CROSSOVER_PROB = CROSSOVER_PROB_HIGH
                MUTATION_PROB  = MUTATION_PROB_HIGH
            else:
                CROSSOVER_PROB = CROSSOVER_PROB_LOW
                MUTATION_PROB  = MUTATION_PROB_LOW
            # end on max iterations - to avoid local maximum
            if generation == MAX_LIFETIME:
                return False
            if not need_to_refresh_hard:
                print("gen: {} best_fitness = {} num_best_in_sol = {} prop {}".format(generation,self.best_fitness,(self.bests_solution[self.best_fitness]), "high" if CROSSOVER_PROB == CROSSOVER_PROB_HIGH else "low") )
        return True

    def random_solution(self):
        return np.random.randint(1, self.size+1, size=(self.size,self.size))

    def new_generation(self):
        global CROSSOVER_PROB, MUTATION_PROB
        new = []

        # add the best solution in prev generation and worst

        if CROSSOVER_PROB == CROSSOVER_PROB_HIGH:
            new.append(self.gen[self.index_worst_sol])
            # force best solution to have given numners
            cost_sol = self.insert_given_numbers(self.gen[self.index_best_sol])
            new.append(cost_sol)
        else:
            new.append(self.gen[self.index_best_sol])
        # add more solution
        while len(new) < POPULATION:
            new_sol = self.choose_random_sol()
            # crossover and mutation
            if randint(0, 100) < CROSSOVER_PROB:
                if CROSSOVER_PROB == CROSSOVER_PROB_HIGH:
                    sol_to_crossover = self.gen[self.index_best_sol]
                else:
                    sol_to_crossover = self.choose_random_sol()
                parents = [new_sol, sol_to_crossover]
                new_sol = self.crossover(parents)
            if randint(0, 100) < MUTATION_PROB:
                new_sol = self.mutation(new_sol)
                new_sol = np.array(new_sol).reshape((self.size,self.size))
            # ids = map(id, new)
            # if id(new_sol) in ids:
            #     continue
            new.append(new_sol)
        return new


    def choose_fitness(self, fitness_func):
        if fitness_func == 'regular':
            return self.fitness()
        if fitness_func == 'darwin':
            return self.darwin_fitness()
        if fitness_func == 'lamarck':
            return self. lamarck_fitness()

    def mutation_line(self, line):
        for i in range(len(line)):
            should_mutate = choices(
                population=[True,False],
                weights=[0.2,0.8])[0]
            if should_mutate:
                new_val = choices(
                    population=[i for i in range (1,self.size+1)],
                    weights=5 * [1])[0]  # returns a 1 element list
                line[i] = new_val
        return line

    def mutation(self, sol):
        # probability = randint(0, 100)
        # if probability <= MUTATION_PROB:
        #     return sol.T
        # return sol

        new = []
        if MUTATION_PROB == MUTATION_PROB_HIGH:
            sol = self.gen[self.index_best_sol]
        for i in range(0, self.size):
            probability = randint(0, 100)
            if probability <= MUTATION_PROB:
                if MUTATION_PROB == MUTATION_PROB_HIGH:
                    line, _ = self.change_line(sol[i])
                    new.append(line)
                else:
                    new.append(self.mutation_line(sol[i]))

            else:
                new.append(sol[i])
        return new

    def choose_random_sol(self):
        scores = list(self.fitness_score.keys())
        if len(scores) == 0:
            return self.gen[self.index_worst_sol]
        scores.sort()
        bad_solutions = int(len(scores) * 0.9)
        good_solutions = len(scores) - bad_solutions

        idx = choices(
            population=scores,
            weights=bad_solutions*[0.05] + good_solutions*[0.95])[0]

        len_solutions = len(self.fitness_score[idx])
        if len_solutions > 1:
            idxx = choices(
                population=[i for i in range(len_solutions)],
                weights=[1] * len_solutions)[0]
            ret_sol = self.fitness_score[idx][idxx]
        else:
            ret_sol = self.fitness_score[idx][0]
        return ret_sol


    def crossover(self, parents):
        # generate a random crossover point
        crossover_point = randint(0, self.size)
        if CROSSOVER_PROB == CROSSOVER_PROB_HIGH:
            crossover_point = randint(0, int(self.size /2) ) #take more from parent 2 - best sol
        # copy everything before this point from parent 1 and after this point from parent 2
        offspring = np.concatenate((parents[0][0:crossover_point],parents[1][crossover_point:]))
        return offspring

    def score(self, solution):
        sum = 0
        for i in range (len(solution)):
            sum += (len(set(solution[i])) - self.size) + 1 # check duplications in rows
            sum += (len(set(solution[:,i])) - self.size) +1 # check duplications in columns
            for j in range(len(solution[i])):
                cell = self.grid.positions[(i,j)]
                cell.val = solution[i][j]

                if (i,j) in self.given_numbers.keys() and cell.val != self.given_numbers[(i,j)]: # punish solutions that ignore given numbers
                    sum -= 99999 # valid solution must contain the right given numbers
                valid = cell.validateInequalities(self.grid)
                sum += 1 if valid else -1
        return sum

    def fitness(self):
        fitness_score = {}
        self.best_fitness = -np.inf
        self.worst_fitness = np.inf
        self.index_best_sol = -1
        self.index_worst_sol = -1
        sol_index = 0
        for sol in self.gen:
            fitness_score_sol = self.score(sol)
            # if fitness_score_sol < -999: #very bad solution
            #     continue
            if not fitness_score_sol in list(fitness_score.keys()): #already has value
                fitness_score[fitness_score_sol] = []
            fitness_score[fitness_score_sol].append(sol) # if I've two solutions with the same score take the last one
            if fitness_score_sol >= self.best_fitness:
                self.best_fitness = fitness_score_sol
                self.index_best_sol = sol_index
            if fitness_score_sol < self.worst_fitness:
                self.worst_fitness = fitness_score_sol
                self.index_worst_sol = sol_index
            sol_index += 1

        return fitness_score

    def optimization(self):

        optimize_gen = []
        for s in self.gen:
            optimize_sol, score = self.optimize_sol(s)
            optimize_gen.append(optimize_sol)

        return optimize_gen

    def change_line(self, line):
        changed = False
        if len(set(line)) < self.size:
            changed = True
            line = np.random.permutation([x for x in range(1,self.size +1 )])

        return line, changed

    def insert_given_numbers(self, sol):
        for key in self.given_numbers.keys():
            sol[key[0]][key[1]] = self.given_numbers[key]
        return sol

    def optimize_sol(self, sol):
        sol = self.insert_given_numbers(sol)
        best_score_sol = self.score(sol)
        num_optimization_allow = self.size
        for i in range (len(sol)):
            if num_optimization_allow == 0:
                break
            line, changed = self.change_line(sol[i]) # rows
            if changed:
                score = self.score(sol)
                if score > best_score_sol:
                    sol[i] = line
                    best_score_sol = score
                    num_optimization_allow -= 1
        if num_optimization_allow > 0:
            for i in range(len(sol)):
                if num_optimization_allow == 0:
                    break
                line, changed = self.change_line(sol[:,i]) # columns
                if changed:
                    score = self.score(sol)
                    if score > best_score_sol:
                        sol[:, i] = line
                        best_score_sol = score
                        num_optimization_allow -= 1
        return sol, best_score_sol

    def darwin_fitness(self):
        optimized_gen = self.optimization()
        optimized_gen, self.gen = self.gen, optimized_gen # swap
        fitness = self.fitness()
        optimized_gen, self.gen = self.gen, optimized_gen # swap
        return fitness

    def  lamarck_fitness(self):
        self.gen = self.optimization()
        fitness = self.fitness()
        return fitness

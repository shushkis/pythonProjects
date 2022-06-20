# imports
import csv
import argparse
import sys
import os.path


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import Hexagonal



PARTY_MAP = {3: 'Labour',
             4: 'Yamina',
             5: 'Yahadot Hatora',
             6: 'The Joint Party',
             7: 'Zionut Datit',
             8: 'Kachul Lavan',
             9: 'Israel Betinu',
             10: 'Licod',
             11: 'Merez',
             12: 'Raam',
             13: 'Yesh Atid',
             14: 'Shas',
             15: 'Tikva Hadasha'}

# utility grid to implement the hexagon grid
grid = np.zeros((9,9), dtype=np.int8)
grid[4] = np.ones(9, dtype=np.int8)
line_of_5 = [0, 0, 1, 1, 1, 1, 1, 0, 0]
line_of_6 = [0, 1, 1, 1, 1, 1, 1, 0, 0]
line_of_7 = [0, 1, 1, 1, 1, 1, 1, 1, 0]
line_of_8 = [1, 1, 1, 1, 1, 1, 1, 1, 0]

grid[0] = grid[-1] = line_of_5
grid[1] = grid[-2] = line_of_6
grid[2] = grid[-3] = line_of_7
grid[3] = grid[-4] = line_of_8



class MyData:
    def __init__(self,infile="Elec_24.csv", num_lines='all'):
        self.raw_data = {}
        self.raw_data_to_test = {}
        self.data_map = {}
        self.infile = infile
        self.num_lines = num_lines
        self.receive_data_from_csv()
        self.fill_hex_from_data()
        self.save_some_for_test()

    def receive_data_from_csv(self):
        lines = []
        with open(self.infile, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                lines.append(row)

        for i in range(1, len(lines)):
            key = int(lines[i][1])
            if key not in self.raw_data.keys():
                self.raw_data[key] = []
            if self.num_lines=='all':
                self.raw_data[key].append([int(x)/ int(lines[i][2]) for x in lines[i][3:]])
            else:
                self.raw_data[key].append(int(lines[i][self.num_lines])/ int(lines[i][2]))

    def fill_hex_from_data(self):
        for i,k in enumerate(self.raw_data.keys()):
            for j in range(len(self.raw_data[k])):
                self.data_map[(i,j)] = k

    def save_some_for_test(self):
        lines = []
        with open(self.infile, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                lines.append(row)

        for i in range(1, int(len(lines) / 10)): # take 10% of data
            rand_key = np.random.randint(1,len(self.raw_data.keys()))
            if rand_key not in self.raw_data_to_test.keys():
                self.raw_data_to_test[rand_key] = []
            self.raw_data_to_test[rand_key].append(self.raw_data[rand_key].pop())


def get_hex_from_grid():
    my_map = {}
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x,y]:
                hex_center = Hexagonal.coor_to_hex(y,x)
                my_map[tuple(hex_center)] = 0
            # else:
            #     my_map[tuple(hex_center)] = 0
    return my_map


class MySom:
    def __init__(self, max_epochs,dataObj,radius=1,learning_rate=0.5,random_order=True):
        self.mydata = dataObj
        self.random_order = random_order

        # Hyperparameter Initialization
        self.x, self.y = 9, 9  # dimensions of Map
        self.radius = radius  # spread of neighborhood
        self.learning_rate = learning_rate  # learning rate
        self.max_epochs = max_epochs  # no of iterations
        self.decay_parameter = self.max_epochs / 2  # decay parameter

        # Distance map and Assigning Weights
        self.vector_size = 13 if self.mydata.num_lines == 'all' else 1
        self.distance_map = np.full((self.x, self.y), np.inf)
        self.weights = np.random.rand(self.x, self.y, self.vector_size)
        self.drawing_map = get_hex_from_grid()

        # Define Neighborhood Region
        self.neighbour_x = np.arange(self.x)
        self.neighbour_y = np.arange(self.y)

    # Function that reduces learning_rate and radius at each iteration
    def decay_learning_rate_sigma(self, iteration):
        learning_rate_ = self.learning_rate / (1 + iteration / self.decay_parameter)
        radius_ = self.radius / (1 + iteration / self.decay_parameter)
        return learning_rate_, radius_

    def get_winner_neuron(self, x):
        s = np.subtract(x, self.weights)  # x - w
        it = np.nditer(self.distance_map, flags=['multi_index'])
        while not it.finished:
            # || x - w || -- distance
            if grid[it.multi_index] == 0:
                it.iternext()
                continue
            self.distance_map[it.multi_index] = np.linalg.norm(s[it.multi_index])
            it.iternext()
        return np.unravel_index(self.distance_map.argmin(), self.distance_map.shape)

    def update_weights(self,win_neuron, inputx, iteration, key):
        # decay learning rate and sigma
        np.seterr(invalid='ignore')

        learning_rate_, radius_ = self.decay_learning_rate_sigma(iteration)

        # get neighborhood around winning cell
        d = 2 * np.pi * (radius_ ** 2)
        ax = np.exp(-1 * np.square(self.neighbour_x - win_neuron[0]) / d)
        ay = np.exp(-1 * np.square(self.neighbour_y - win_neuron[1]) / d)

        neighborhood = np.outer(ax, ay)

        it = np.nditer(neighborhood, flags=['multi_index'])
        while not it.finished:
            if grid[it.multi_index] == 0:
                it.iternext()
                continue
            self.weights[it.multi_index] += learning_rate_ * neighborhood[it.multi_index] * (
                        inputx - self.weights[it.multi_index])
            # my_map[it.multi_index] = key
            it.iternext()


    def train(self):
        # Training model: Learning Phase
        data_ = []
        for k in self.mydata.raw_data.keys():
            for v in self.mydata.raw_data[k]:
                data_.append(v)
        idx = 0
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            np.random.shuffle(data_)
            if self.random_order:
                idx = np.random.randint(0, len(data_))
            else:
                if idx == len(self.data_):
                    idx = 0
            win_neuron = self.get_winner_neuron(data_[idx])
            key = 0
            for k in self.mydata.raw_data.keys():
                if data_[idx] in self.mydata.raw_data[k]:
                    key = k
                    break
            hex_coor = Hexagonal.coor_to_hex(win_neuron[0], win_neuron[1])
            try:
                # if (my_map[tuple(hex_coor)]):
                self.drawing_map[tuple(hex_coor)] = key
            except:
                pass
            self.update_weights(win_neuron, data_[idx], epoch,key)

            idx +=1
            if epoch == 1 or epoch == 100 or epoch == 1000 or epoch == 10000 or epoch == 50000:
                # continue
                self.plot(epoch)
                try:
                    temp_arr = self.weights[np.unravel_index(win_neuron, self.weights.shape[:-1])]
                    quantization_error = np.linalg.norm(data_ - temp_arr[0], axis=1).mean()
                    print('\n quantization error:', quantization_error)
                except:
                    pass




    def test(self):
        total_cnt = 0
        right_cnt = 0
        almost_right = 0
        for k in self.mydata.raw_data_to_test.keys():
            for v in self.mydata.raw_data_to_test[k]:
                win_neuron = self.get_winner_neuron(v)
                hex_coor = Hexagonal.coor_to_hex(win_neuron[0], win_neuron[1])
                pred_key = self.drawing_map[tuple(hex_coor)]
                total_cnt += 1
                right_cnt += k == pred_key
                if abs(k-pred_key) == 1:
                    almost_right += 1
                print("tested key: {} predicted key: {}".format(k, pred_key))
        print("I was right {} out of {} tries which means {}% \nand almost right {} times which menas {}%".format(right_cnt, total_cnt, (right_cnt/total_cnt * 100), almost_right, (almost_right/total_cnt * 100)))

    def close_event(self):
        plt.close() #timer calls this function after interval seconds and closes the window


    def plot(self, epoch):

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1 = Hexagonal.plot_hex(ax1, self.mydata.data_map)
        ax2 = Hexagonal.plot_hex(ax2, self.drawing_map)

        which_party = self.mydata.num_lines if self.mydata.num_lines == 'all' else PARTY_MAP[self.mydata.num_lines]
        fig.suptitle('Training data         After ' + str(epoch) + ' iterations\n for ' + which_party +' Party')
        timer = fig.canvas.new_timer(interval=3000)  # creating a timer object and setting an interval of 3 seconds
        timer.add_callback(self.close_event)
        if not epoch == self.max_epochs:
            timer.start()
        norm = colors.Normalize(vmin=0, vmax=10)
        sm = plt.cm.ScalarMappable(cmap=Hexagonal.colors,norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        plt.show()
        # plt.pause(0.1)
        # plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='some arguments to pass via cml')
    parser.add_argument("-f", "--file",  default="./Elec_24.csv", required=False, help="CSV to work with")
    parser.add_argument("-e", "--epochs", type=int, default=10000, required=False, help="num of epochs to do")
    parser.add_argument("--learning_rate" , type=float, default= 0.5, required=False)
    parser.add_argument("--radius", type=float, default=0.5, required=False, help="spread of neighborhood")


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--random_order", type=str2bool, default=True, required=False, help="Train in random order")


    # def party_map(val):
    #     if val == None:
    #         return 'all'
    #     return PARTY_MAP.get(int(val))
    parser.add_argument("--party", required=False, type=int, help=", ".join((str(x) for x in PARTY_MAP.items())), choices=PARTY_MAP)
    parser.print_help()
    args = parser.parse_args()
    args.party = 'all' if args.party == None else args.party
    print("\nchosen args: ", args)
    print("\nEpochs:\n")

    # check that file exists and csv
    if not os.path.exists(args.file) or not args.file.endswith("csv"):
        print(args.file, "not exists or not csv")
        sys.exit()

    data_obj = MyData(args.file, args.party)
    som_obj = MySom(args.epochs, data_obj)
    som_obj.train()
    som_obj.test()




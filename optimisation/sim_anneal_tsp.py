import numpy as np
import matplotlib.pyplot as plt


class SimAnneal(object):
    """ Simulated annealing

        Travelling salesman

    """

    def __init__(self, n_cities=10):
        # super(SimAnneal, self).__init__()
        self.n = n_cities
        self.cords = None  # cordinates of the cities
        self.cur_sol = None
        self.cur_cost = None
        self.cost_history = []
        #
        self.T = 1000  # temperature
        self.MAX_STEPS = int(1e6)
        self.T_STOP = 1e-6
        self.T_DECAY = 0.9999
        #
        self.ax = None

    def generate_cities(self, cities_type='random'):
        if cities_type == 'grid':
            if self.n >= 20:
                raise Exception
            tmp = np.linspace(0, 1, self.n)
            x, y = np.meshgrid(tmp, tmp)
            x = x.flatten()
            y = y.flatten()
            self.n = x.size
        else:
            x = np.random.rand(self.n)
            y = np.random.rand(self.n)
        self.cords = np.c_[x, y]

    def initialize(self):
        self.cur_sol = np.random.permutation(self.n)

    def get_neighbours(self):
        # TODO: get neighbouring permutations
        '''
        #  Stratagies
        #1 swap two random cities
        #2 swap one random sub-paths ab | cdef | g --> ab | fedc | g
        '''
        a, b = np.sort(np.random.choice(self.n, 2, replace=False))
        subpath = self.cur_sol[a:b]
        flipped = np.flip(subpath)
        new_sol = self.cur_sol.copy()
        new_sol[a:b] = flipped
        return new_sol

    def cost(self, city_ids):
        # total Euclidean distance
        distance = 0
        cords = self.cords[city_ids]
        for old, new in zip(cords[:-1], cords[1:]):
            diff = old - new
            tmp = diff[None, :] @ diff[:, None]
            distance += np.sqrt(tmp)
        first = self.cords[0]
        last = self.cords[-1]
        diff = last - first
        distance += np.sqrt(diff[None, :] @ diff[:, None])
        return distance

    def solve(self, if_display=True):
        old_cost = self.cost(self.cur_sol)
        self.cur_cost = old_cost
        self.cost_history.append(old_cost[0][0])
        self.display()
        # for i in range(self.MAX_STEPS):
        i = 0
        while self.T >= self.T_STOP:
            # self.T = (1 - (i + 1) / self.MAX_STEPS)
            # if self.T < 1e-5:
            #     return
            new_sol = self.get_neighbours()
            new_cost = self.cost(new_sol)
            #
            if new_cost < old_cost:
                self.cur_sol = new_sol
                self.cur_cost = new_cost
                old_cost = new_cost
            else:
                delta = new_cost - old_cost
                prob = np.exp(-delta / (self.T))
                if prob >= np.random.uniform():
                    self.cur_sol = new_sol
                    self.cur_cost = new_cost
                    old_cost = new_cost
            self.T *= self.T_DECAY
            i += 1
            #print(i, self.T, old_cost, new_cost)

            if i % 100 == 0:
                self.cost_history.append(old_cost[0][0])

            # draw solutions
            if if_display and (i % 1000) == 0:
                self.display()

    def display(self):
        if not self.ax:
            plt.ion()
            _, self.ax = plt.subplots()
        self.ax.cla()
        cords = self.cords[self.cur_sol]
        first = cords[0][None, :]  # looooopy
        cords = np.r_[cords, first]
        self.ax.plot(*cords.T, '.-', lw=0.75, alpha=0.95, ms=10)
        txt = f'T = {self.T:2.5f} cost = {self.cur_cost[0][0]:2.3f}'
        self.ax.set_title(txt)
        plt.draw()
        plt.pause(0.01)

    def demo(self):
        #
        self.generate_cities()
        # initial guess
        self.initialize()
        self.solve()

        plt.figure()
        plt.plot(self.cost_history)
        plt.show()


if __name__ == "__main__":
    n_cities = 10
    tsp = SimAnneal(n_cities)
    tsp.demo()

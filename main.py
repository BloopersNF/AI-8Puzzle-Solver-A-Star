import numpy as np
from queue import PriorityQueue
from scipy.signal import convolve2d as conv2


class State:
    def __init__(self, parent=None, matrix=None):
        self.matrix = matrix
        self.parent = parent

        self.g = 0
        self.f = 0

    def __eq__(self, other):
        return len(self.matrix[(np.where(self.matrix != other.matrix))]) == 0
    
    def __lt__(self, other):
        return self.f < other.f
    
    def show(self):
        for i in self.matrix:
            print(i)
        print()

def play(state, block):
    
    newM = state.matrix.copy()

    newM[np.where(state.matrix == block)] = 9
    newM[np.where(state.matrix == 9)] = block

    return State(matrix=newM)


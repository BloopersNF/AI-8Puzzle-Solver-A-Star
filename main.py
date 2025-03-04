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

def plays(state):
    
    adj = np.array([[0,1,0],[1,0,1],[0,1,0]])
    blank = state.matrix==9
    conv = conv2(adj, blank, 'same')

    return state.matrix[np.where(conv)]

matriz1 = State(matrix=np.array([[1,2,3],[4,5,6],[7,8,9]]))
matriz2 = State(matrix=np.array([[1,2,3],[4,5,6],[7,9,8]]))

print(matriz1.matrix == 3)
print(matriz1.matrix == 0)
print(matriz1.matrix == 9)

matriz2.show()

matriz2 = play(matriz2, 8)
matriz2.show()
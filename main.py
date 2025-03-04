import numpy as np
from queue import PriorityQueue
from scipy.signal import convolve2d as conv2
from time import sleep

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

def AStar(state, goal, heuristic):

    cs = set()

    pq = PriorityQueue()
    state.f = 0
    state.parent = None

    pq.put((state.f, state))

    while pq is not pq.empty():
        s = pq.get()[1]

        if s == goal:
            return s
        
        if tuple(s.matrix.flatten()) in cs:
            continue

        cs.add(tuple(s.matrix.flatten()))

        for p in plays(s):
            ns = play(s, p)
            ns.g = s.g + 1
            ns.parent = s
            ns.f = ns.g + heuristic(ns)
            pq.put((ns.f, ns))
    return state


def hamming(state):
        goal = np.array([[1,2,3],[4,5,6],[7,8,9]])
        out = len(state.matrix[np.where(state.matrix != goal)]) - 1
        return max(out, 0)

def manhattan(state):
    return 0


def reconstruct(state):
    if state.parent is not None:
        reconstruct(state.parent)
    print(state.matrix)


goal = State(matrix=np.array([[1,2,3],[4,5,6],[7,8,9]]))
matriz = State(matrix=np.array([[4,6,7],[9,5,8],[2,1,3]]))

result = AStar(matriz, goal, hamming)

print("matriz inicial:")
print(matriz.matrix)
print('Encontrando solução...')
print()

reconstruct(result)
print(f'Objetivo encontrado em {result.g} movimentos')




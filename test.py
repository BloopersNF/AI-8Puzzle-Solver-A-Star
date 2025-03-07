import numpy as np
from queue import PriorityQueue
import csv
from scipy.signal import convolve2d as conv2
from time import perf_counter as counter

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

def invCount(matrix):
    arr = matrix.flatten()
    arr = arr[arr != 9]
    inv = sum(1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j])
    return inv

def hamming(state):
    goal = np.array([[1,2,3],[4,5,6],[7,8,9]])
    out = len(state.matrix[np.where(state.matrix != goal)]) - 1
    return max(out, 0)

def manhattan(state):
    dist = 0
    goal = np.array([[1,2,3],[4,5,6],[7,8,9]])
    for i in range(state.matrix.shape[0]):
        for j in range(state.matrix.shape[1]):
            block = state.matrix[i,j]
            if block != 9:
                gPos = np.argwhere(block == goal)[0]
                dist += abs(i - gPos[0]) + abs(j - gPos[1])
    return dist

def AStar(state, goal, heuristic, tie_breaker):
    stateCount = 0
    cs = set()
    pq = PriorityQueue()
    state.f = 0
    state.parent = None
    pq.put((state.f, heuristic(state), state.g, state))

    while not pq.empty():
        s = pq.get()[3]
        stateCount += 1
        
        if s == goal:
            return s, stateCount
        
        if (invCount(s.matrix) % 2) != 0:
            raise ValueError("Matrix not solvable")
        
        if tuple(s.matrix.flatten()) in cs:
            continue

        cs.add(tuple(s.matrix.flatten()))

        for p in plays(s):
            ns = play(s, p)
            ns.g = s.g + 1
            ns.parent = s
            ns.f = ns.g + heuristic(ns)
            
            if tie_breaker == 'min_g':
                pq.put((ns.f, ns.g, heuristic(ns), ns))
            elif tie_breaker == 'fifo':
                pq.put((ns.f, stateCount, heuristic(ns), ns))
            else:
                pq.put((ns.f, heuristic(ns), ns.g, ns))
    return None, stateCount

def run_experiment():
    goal = State(matrix=np.array([[1,2,3],[4,5,6],[7,8,9]]))
    matriz = State(matrix=np.array([[4,6,7],[9,5,8],[2,1,3]]))
    heuristics = [('Hamming', hamming), ('Manhattan', manhattan)]
    tie_breakers = ['min_h', 'min_g', 'fifo']

    results = []
    for heuristic_name, heuristic in heuristics:
        for tie in tie_breakers:
            start = counter()
            result, stateCount = AStar(matriz, goal, heuristic, tie)
            end = counter()
            results.append([heuristic_name, tie, result.g if result else 'N/A', stateCount, abs(end - start)])
    
    with open('astar_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Heuristic', 'Tie Breaker', 'Movements', 'States Explored', 'Time (s)'])
        writer.writerows(results)
    
    print("Results saved to astar_results.csv")

run_experiment()

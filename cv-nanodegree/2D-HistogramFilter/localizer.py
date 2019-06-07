# import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []
    for i in range(len(grid)):
        tmp_list = list()
        for j in range(len(grid[i])):
            hit = (color == grid[i][j])
            tmp_list.append(beliefs[i][j] * (hit * p_hit + (1-hit) * p_miss))
            
        new_beliefs.append(tmp_list)

    total_sum = sum([sum(x) for x in new_beliefs])
    
    for i in range(len(new_beliefs)):
        for j in range(len(new_beliefs[i])):
            new_beliefs[i][j] = new_beliefs[i][j] / total_sum

    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(height)] for j in range(width)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy ) % width
            new_j = (j + dx ) % height
            new_G[int(new_i)][int(new_j)] = cell
            #import pdb; pdb.set_trace()
            
    return blur(new_G, blurring)
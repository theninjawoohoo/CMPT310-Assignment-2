# Author: Andy Wu
# SFU Number: 301308902

from search import *
import random
import time
import copy

# --------------------------------------------
# BFS Modded - Return node and totalnumberofnodes
def modded_best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    totalnumbernodes = 0
    # Added some code to count the fronter popping
    while frontier:
        node = frontier.pop()
        totalnumbernodes = totalnumbernodes + 1
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node, totalnumbernodes
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

    return None, 0

# --------------------------------------------
# A* modded
def modded_astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return modded_best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

# Question 1 - Helper Functions
def make_rand_8puzzle():
    init_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    random.shuffle(init_state)
    tuple_state = tuple(init_state)
    #  init_state must be a tuple according to assignment information
    new_puzzle = EightPuzzle(tuple_state)

    # if the puzzle is not solvable repeat and keep shuffling the array
    while (new_puzzle.check_solvability(tuple_state) == False):
        random.shuffle(init_state)
        tuple_state = tuple(init_state)
        new_puzzle = EightPuzzle(tuple_state)
    display(new_puzzle.initial)
    return new_puzzle

# Print in python by default prints new lines, so add end='' to prevent it from happening
def display(state):
    counter = 0
    while counter < len(state):
        if state[counter] == 0:
            print("* ", end='')
        else:
            print(state[counter], end='')
            print(" ", end='')
        if (counter + 1) % 3 == 0:
            print("")
        counter = counter + 1


# make_rand_8puzzle()
# --------------------------------------------
# Question 2 - Comparing Algorithms
def astarwithmisplacedh(unsolvedpuzzle):
    start_time = time.time()

    # H and time do not need to be passed. it seems the h by default will use the puzzle's heuristic
    popped, counter = modded_astar_search(unsolvedpuzzle)
    elapsed_time = time.time() - start_time

    print(f'astar with misplaced heuristic took: {elapsed_time}s')
    print(f'astar with misplaced heuristic total frontier nodes popped: {counter} nodes')
    print(f'astar with misplaced heuristic took number of moves: {len(popped.solution())} nodes')

# O(n) calculation to obtain the heuristic value
def mymanhattan(unsolvedpuzzle):
    # Taken from the search.py
    # Build a dictionary of coordinates for the solve state and a 2D version of the initial state
    state = [[], [], []]

    # solve state according to search.py is
    # 1 2 3
    # 4 5 6
    # 7 8 0
    solve_state = {0:[2, 2], 1:[0, 0], 2:[1, 0], 3:[2, 0], 4:[0, 1], 5:[1, 1], 6:[2, 1], 7:[0, 2], 8:[1, 2]}
    counter = 0
    for y in range(0, 3):
        for x in range(0, 3):
            list_y = state[y]
            list_y.append(unsolvedpuzzle.state[counter])
            counter = counter + 1

    manhattan_sum = 0

    for y in range(0, 3):
        for x in range(0, 3):
            current_tile = state[y][x]
            coordinates = solve_state[current_tile]
            manhattan_sum = abs(coordinates[0] - x) + abs(coordinates[1] - y) + manhattan_sum

    return manhattan_sum

# a star with manhattan
def astarwithmanhattan(unsolvedpuzzle):
    start_time = time.time()
    popped, counter = modded_astar_search(unsolvedpuzzle, h=mymanhattan)
    elapsed_time = time.time() - start_time
    print(f'astar with manhattan heuristic took: {elapsed_time}s')
    print(f'astar with manhattan heuristic total frontier nodes popped: {counter} nodes')
    print(f'astar with manhattan heuristic took hnumber of moves: {len(popped.solution())} nodes')


# find max algo between manhattan and misplaced tile
def maxmisplaced(unsolvedpuzzle):
    # We need to recreate the puzzle object to object the misplaced tile heuristic
    # The heuristic cannot be obtained from the unsolved puzzle node
    misplacetilepuzzle = EightPuzzle(unsolvedpuzzle.state)
    max_h = max(mymanhattan(unsolvedpuzzle), misplacetilepuzzle.h(unsolvedpuzzle))
    return max_h

# find max of astar with manhattan
def astarwithhandmanhattan(unsolvedpuzzle):
    start_time = time.time()
    popped, counter = modded_astar_search(unsolvedpuzzle, h=maxmisplaced)
    elapsed_time = time.time() - start_time
    print(f'astar with max of manhattan heuristic and misplaced tile heuristic took: {elapsed_time}s')
    print(f'astar with max of manhattan heuristic total frontier nodes popped: {counter} nodes')
    print(f'astar with max of manhattan heuristic and misplaced tile hnumber of moves: {len(popped.solution())} nodes')

# Main Loop.
def compare_algo():
    # Generate 10 Puzzles
    counter = 0
    number_of_puzzles = 10
    while counter != number_of_puzzles:
        print(f'Solving Puzzle Number:{counter + 1}')
        print("Here is the puzzle...")

        max_puzzle = make_rand_8puzzle()
        manhattan_puzzle = copy.deepcopy(max_puzzle)
        astar_puzzle = copy.deepcopy(max_puzzle)

        astarwithmisplacedh(astar_puzzle)
        astarwithmanhattan(manhattan_puzzle)
        astarwithhandmanhattan(max_puzzle)

        counter = counter + 1

compare_algo()
# --------------------------------------------
#Question 3 Duck Puzzle
class DuckPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 2x4x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square == 0:
            possible_actions.remove('UP')
            possible_actions.remove('LEFT')
        elif index_blank_square == 8:
            possible_actions.remove('DOWN')
            possible_actions.remove('RIGHT')
        elif index_blank_square == 1 or index_blank_square == 5:
            possible_actions.remove('UP')
            possible_actions.remove('RIGHT')
        elif index_blank_square == 2 or index_blank_square == 6:
            possible_actions.remove('LEFT')
            possible_actions.remove('DOWN')
        elif index_blank_square == 4:
            possible_actions.remove('UP')
        elif index_blank_square == 7:
            possible_actions.remove('DOWN')


        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)
        delta = {}

        # Duck Puzzle has various delta cases
        if blank == 0:
            delta = {'UP': 0, 'DOWN': 2, 'LEFT': 0, 'RIGHT': 1}
        elif blank == 1:
            delta = {'UP': 0, 'DOWN': 2, 'LEFT': -1, 'RIGHT': 0}
        elif blank == 2:
            delta = {'UP': -2, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 1}
        elif blank == 3:
            delta = {'UP': -2, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        else:
            delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}

        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

# Print in python by default prints new lines, so add end='' to prevent it from happening (duck version)
def duck_display(state):
    counter = 0
    while counter < len(state):
        if state[counter] == 0:
            print("* ", end='')
        else:
            print(state[counter], end='')
            print(" ", end='')
        if counter == 1 or counter == 5:
            print("")
        if counter == 5:
            print("  ", end='')
        counter = counter + 1
    print("")

# Duck Puzzle
def make_rand_duck_puzzle():
    init_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    random_duck_puzzle = DuckPuzzle(init_state)
    counter = 0
    new_state = []
    # 500 Shuffles starting from random state
    while counter < 500:
        valid_actions = random_duck_puzzle.actions(random_duck_puzzle.initial)
        chosen_action = random.choice(valid_actions)
        new_state = random_duck_puzzle.result(random_duck_puzzle.initial, chosen_action)
        random_duck_puzzle = DuckPuzzle(new_state)
        counter = counter + 1

    duck_display(new_state)
    return random_duck_puzzle

# Duck A Star w/ Misplaced H
def duckastarh(unsolvedduckpuzzle):
    start_time = time.time()

    # H and time do not need to be passed. it seems the h by default will use the puzzle's heuristic
    popped, counter = modded_astar_search(unsolvedduckpuzzle)
    elapsed_time = time.time() - start_time

    print(f'duck puzzle astar with misplaced heuristic took: {elapsed_time}s')
    print(f'duck puzzle astar with misplaced heuristic total frontier nodes popped: {counter} nodes')
    print(f'duck puzzle astar with misplaced heuristic took number of moves: {len(popped.solution())} nodes')


# My manhatten for duck puzzle
def duck_manhatten(unsolvedduckpuzzle):
    # Taken from the search.py
    # Build a dictionary of coordinates for the solve state and a 2D version of the initial state
    state = [[], [], [-100]]
    counter = 0
    for y in range(0, 3):
        if y == 0:
            for x in range(0, 2):
                list_y = state[y]
                list_y.append(unsolvedduckpuzzle.state[counter])
                counter = counter + 1
        elif y == 1:
            for x in range(0, 4):
                list_y = state[y]
                list_y.append(unsolvedduckpuzzle.state[counter])
                counter = counter + 1
        elif y == 2:
            for x in range(1, 4):
                list_y = state[y]
                list_y.append(unsolvedduckpuzzle.state[counter])
                counter = counter + 1
    # solve state according to search.py is
    # 1 2
    # 3 4 5 6
    #   7 8 0
    solve_state = {0: [3, 2], 1: [0, 0], 2: [1, 0], 3: [0, 1], 4: [1, 1], 5: [2, 1], 6: [3, 1], 7: [1, 2], 8: [2, 2]}

    manhattan_sum = 0

    for y in range(0, 3):
        if y == 0:
            for x in range (0, 2):
                current_tile = state[y][x]
                coordinates = solve_state[current_tile]
                manhattan_sum = abs(coordinates[0] - x) + abs(coordinates[1] - y) + manhattan_sum
        elif y == 1:
            for x in range (0, 4):
                current_tile = state[y][x]
                coordinates = solve_state[current_tile]
                manhattan_sum = abs(coordinates[0] - x) + abs(coordinates[1] - y) + manhattan_sum
        elif y == 2:
            for x in range (1, 4):
                current_tile = state[y][x]
                coordinates = solve_state[current_tile]
                manhattan_sum = abs(coordinates[0] - x) + abs(coordinates[1] - y) + manhattan_sum
    return manhattan_sum

#For duck manhattan
def astarwithduckmanhattan(unsolvedduckpuzzle):
    start_time = time.time()
    popped, counter = modded_astar_search(unsolvedduckpuzzle, h=duck_manhatten)
    elapsed_time = time.time() - start_time
    print(f'duck astar with manhattan heuristic took: {elapsed_time}s')
    print(f'duck astar with manhattan heuristic total frontier nodes popped: {counter} nodes')
    print(f'duck astar with manhattan heuristic took hnumber of moves: {len(popped.solution())} nodes')

# find max algo between manhattan and misplaced tile (duck version)
def duckmaxmisplaced(unsolvedpuzzle):
    # We need to recreate the puzzle object to object the misplaced tile heuristic
    # The heuristic cannot be obtained from the unsolved puzzle node
    misplacetilepuzzle = DuckPuzzle(unsolvedpuzzle.state)
    max_h = max(duck_manhatten(unsolvedpuzzle), misplacetilepuzzle.h(unsolvedpuzzle))
    return max_h

# find max of astar with manhattan (duck version)
def duckastarwithhandmanhattan(unsolvedpuzzle):
    start_time = time.time()
    popped, counter = modded_astar_search(unsolvedpuzzle, h=duckmaxmisplaced)
    elapsed_time = time.time() - start_time
    print(f'duck astar with max of manhattan heuristic and misplaced tile heuristic took: {elapsed_time}s')
    print(f'duck astar with max of manhattan heuristic total frontier nodes popped: {counter} nodes')
    print(f'duck astar with max of manhattan heuristic and misplaced tile hnumber of moves: {len(popped.solution())} nodes')


# duck puzzle general algorithm
def duck_puzzle():
    # Generate 10 Puzzles
    counter = 0
    number_of_puzzles = 10
    while counter != number_of_puzzles:
        print(f'Solving Puzzle Number:{counter + 1}')
        print("Here is the puzzle...")

        maxduckpuzzle = make_rand_duck_puzzle()
        manhattanduck_puzzle = copy.deepcopy(maxduckpuzzle)
        astarduckpuzzle = copy.deepcopy(maxduckpuzzle)

        duckastarh(astarduckpuzzle)
        astarwithduckmanhattan(manhattanduck_puzzle)
        duckastarwithhandmanhattan(maxduckpuzzle)

        counter = counter + 1

duck_puzzle()
# --------------------------------------------
import random
from csp import *
from a2_q1 import *
from a2_q2 import *
import random
import time


# Modified Code from the aima library
class my_CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

        # Added the unassigned variable and my personal information
        self.uassigns = 0
        self.number_of_prunes = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

            # When the conditional occurs increment uassign
            self.uassigns += 1

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        self.number_of_prunes += 1
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# Let Domain be all possible groups anb individual can join
# Let the keys in this dict be the people and the values be possible groups
# We will build the domain from the ground up
# 1st iteration: each person will be only able to join 1 group.
# x iteration: each person will be only able to join x groups.
# A small helper function to populate the domain.
# group_list = [0]
# Example output... {0:[0, 1], 1:[0, 1] 2:[0, 1]}
# This means person 1 2 3 can only join group 0.
def expand_domain(group_num, group_list):
    domains = {}
    for key in range(group_num):
        domains.setdefault(key, group_list)
    return domains


# Code borrowed from a2_q2
def min_number_of_teams(csp_sol):
    group_dict = {}

    # Init group dict. Set the groups
    for group in csp_sol:
        group_dict[csp_sol[group]] = []

    for group in csp_sol:
        group_dict[csp_sol[group]].append(group)

    return len(group_dict), group_dict


# Let's deal with one graph at a time.
# Example Graph: {0: [1, 2], 1: [0], 2: [0], 3: []}
def run_q3_one_graph(friend_group):
    # Create the list of variables needed for the CSP
    # Variables = People
    variable = friend_group.keys()

    # Recorded variables as requested on the assignment
    number_of_teams = 0
    elapsed_time = 0
    csp_assign = 0
    csp_unassign = 0

    # Saving number of prunes
    number_of_prunes = 0

    # Algorithm according to course page
    # Throw necessary information into a CSP then apply AC3 to limit the scope of the domain,
    # cut the number of domains we need to search
    # Then backtrack solve for all domains until a solution is found. Since our domain is built starting with 1 group
    # We are guaranteed the min number of teams needed
    # Our worse case is 30 groups assuming everyone is a friend of each other
    possible_group_domain = []

    # print(friend_group)
    solution = None
    group_dict = None
    # Start timer for algo
    start_time = time.time()
    for group_num in range(len(variable)):
        possible_group_domain.append(group_num)
        domain = expand_domain(len(variable), possible_group_domain)

        # We create the csp to cut down on the huge domain scope
        # https://www.quora.com/What-does-the-arc-consistency-algorithm-AC3-do
        csp_friend = my_CSP(variable, domain, friend_group, different_values_constraint)
        AC3(csp_friend)

        # From csp.py line 775

        solution = backtracking_search(csp_friend, mrv, lcv, forward_checking)
        csp_assign += csp_friend.nassigns
        csp_unassign += csp_friend.uassigns

        # Once we find the one possible solution we stop the timer and stop solving the expanding domains
        if solution is not None:
            elapsed_time = time.time() - start_time
            number_of_teams, group_dict = min_number_of_teams(solution)
            number_of_prunes = csp_friend.number_of_prunes
            break

    return elapsed_time, csp_assign, csp_unassign, number_of_teams, number_of_prunes, solution, group_dict


# Helper function to generate graphs
def generate_graphs(n, graphs):
    for i in range(n):
        graph = [rand_graph(0.1, 31), rand_graph(0.2, 31), rand_graph(0.3, 31),
                 rand_graph(0.4, 31), rand_graph(0.5, 31), rand_graph(0.6, 31)]
        graphs.append(graph)


def print_results(graph):
    elapsed_time, csp_assign, csp_unassign, number_of_teams, number_of_prunes, solution, group_dict = run_q3_one_graph(
        graph)
    print("===========================================================")
    print("Graph: ", graph)
    print("Time Elapsed: ", elapsed_time)
    print("Number of times CSP variables were assigned: ", csp_assign)
    print("Number of times CSP variables were unassigned: ", csp_unassign)
    print("Aprox Min Number of teams needed: ", number_of_teams)
    print("Number of prunes in CSP: ", number_of_prunes)
    print("Solution: ", solution)
    print("Solution Reformatted: ", group_dict)
    print("===========================================================")


def run_q3():
    graphs = []
    generate_graphs(5, graphs)

    for i in range(5):
        print(f'Trial {i + 1} Running Graph with p:0.1, n:31...')
        print_results(graphs[i][0])
        print(f'Trial {i + 1} Running Graph with p:0.2, n:31...')
        print_results(graphs[i][1])
        print(f'Trial {i + 1} Running Graph with p:0.3, n:31...')
        print_results(graphs[i][2])
        print(f'Trial {i + 1} Running Graph with p:0.4, n:31...')
        print_results(graphs[i][3])
        print(f'Trial {i + 1} Running Graph with p:0.5, n:31...')
        print_results(graphs[i][4])
        print(f'Trial {i + 1} Running Graph with p:0.6, n:31...')
        print_results(graphs[i][5])


run_q3()

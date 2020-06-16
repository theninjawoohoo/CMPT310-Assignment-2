# Author: Andy Wu
# SFU Number: 301308902

import random
# --------------------------------------------

# 1. Initialize dictionary
# 2. For each number in i in the relationships dict use the p to randomly assign friends to i
# 3. Add it to other friend if not accounted for
def rand_graph(p, n):
    relationships = {}
    # Initialize dictionary
    for i in range(n):
        relationships[i] = []

    # Iteration through the dict
    for i in range(n):
        for j in range(n):
            if i != j:
                if random.random() < p:
                    # Ensure that duplicates don't occur
                    if j not in relationships[i]:
                        relationships[i].append(j)
                    if i not in relationships[j]:
                        relationships[j].append(i)

    return relationships

# print(rand_graph(0.1, 10))

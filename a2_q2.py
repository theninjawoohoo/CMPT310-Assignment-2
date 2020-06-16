# Author: Andy Wu
# SFU Number: 301308902


# Returns True if the given CSP solution dictionary csp_sol
# satisfies all the constraints in the friendship graph, and False otherwise.

# 1. Convert the csp_sol into a dict, ie groups that have members {0:[0,1]} (group 0 has members 0 and 1)
# 2. Iteratively go through the dict using
def check_teams(graph, csp_sol):
    group_dict = {}

    # Init group dict. Set the groups
    for group in csp_sol:
        group_dict[csp_sol[group]] = []

    for group in csp_sol:
        group_dict[csp_sol[group]].append(group)

    # print(group_dict)

    # Ensure that none of the group members are friends with each other
    for group in group_dict:
        group_members = group_dict[group]

        # Obtain the friends of that member in the group
        for member in group_members:
            friend_list = graph[member]

            # If any any of the friends are in the group it's not compatible
            for friend in friend_list:
                if friend in group_members:
                    return False

    return True


# test
# graph = {0: [1, 2], 1: [0], 2: [0], 3: []}
# csp_sol = {0:0, 1:0, 2:1, 3:1}
# print(check_teams(graph,csp_sol))
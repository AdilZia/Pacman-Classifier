# traceAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Directions
import random
import api
import os

# Calculate the distance between two point


def hv_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


# Following this class, we create the object(node) for using a* algorithm
class Node:

    def __init__(self, current_point):
        # node's parent node
        self.parent = None
        # node's coordinate
        self.node = current_point
        # g is the cost of the path from the start node to here
        self.g = 0
        # h is a heuristic that estimates the cost of the cheapest path from
        # here to the goal.
        self.h = 0
        # f = g + h
        self.f = 0
        # node's children's node
        self.children = []

    # method to calculate h cost of the node
    def calculcate_h(self, end_point):
        self.h = hv_distance(self.node, end_point)

    # method to calculate g cost of the node
    def calculate_g(self):
        self.g = self.parent.g + hv_distance(self.node, self.parent.node)

    # method to calculate f cost of the node
    def calculate_f(self):
        self.f = self.h + self.g


# execute a* algorithm
# Input:
#     current_node: the object Node
#     end_point: tuple, (x, y), the coordinate of the destination
#     obstacles: the list of coordinates (x, y), which is not allowed to pass
#     open_set: the set of Node's objects, which are candidate we are going to find deeply
#     closed_set: the set of Node's objects, which are selected from open_set
# Output:
#     the set of Node's objects which we will get optimal path from
def A_Star(current_node, end_point, obstacles, open_set, closed_set):
    # extract the current coordinate from the object Node
    current_point = current_node.node
    # computing the neighbors of the current
    candidates = list(set([(current_point[0] + i, current_point[1] + j) for i in [-1, 0, 1]
                           for j in [-1, 0, 1] if abs(i) + abs(j) == 1]) - set(obstacles))
    # find the boundary of the map
    boundary = [(min(i), max(i)) for i in zip(*obstacles)]
    # filter some candidates which are not in the map
    candidates = [i for i in candidates if i[0] > boundary[0][0] and i[0]
                  < boundary[0][1] and i[1] > boundary[1][0] and i[1] < boundary[1][1]]

    # Don't need to find the point which has been in the close list
    candidates = set(candidates) - set([i.node for i in closed_set])

    # If it meet the destination or there is no other point in the open set,
    # it stop finding deeply.
    # if current_point != end_point and len(open_set) > 0:
    if len(open_set) > 0:
        try:
            # looking for each candidate
            for element in candidates:
                # make it the object of Node
                candidate_node = Node(element)
                # set its parent
                candidate_node.parent = current_node
                # calculate its h
                candidate_node.calculcate_h(end_point)
                # calculate its g
                candidate_node.calculate_g()
                # calculate its f
                candidate_node.calculate_f()

                # if a node with the same position as candidate is in the open set which has a lower f than candidate, skip this candidate
                # if a node with the same position as candidate is in the
                # closed set which has a lower f than successor, skip this
                # candidate
                criterion = [i for i in open_set.union(closed_set) if (
                    i.f < candidate_node.f) and (i.node == candidate_node.node)]
                # Otherwise, add this candidate to th open set and to the
                # children node of the current node
                if len(criterion) == 0:
                    open_set.add(candidate_node)
                    current_node.children.append(candidate_node)

            # remove the current node from the open set
            open_set.remove(current_node)
            # add the current node into the closed set
            closed_set.add(current_node)
            # if arriving end_point, return closed_set
            if current_node.node == end_point:
                return closed_set
            # find the node in the open set with the minimum f
            next_node = min(open_set, key=lambda i: i.f)
            return A_Star(next_node, end_point, obstacles, open_set, closed_set)
        # If there is some error, it will return the closed set
        except:
            return closed_set
    # If it meet the destination or there is no point in the open set, it
    # return the closed set.
    else:
        return closed_set


# Find the path from the result of A_Star/the closed set.
# Input:
#    end_point: tuple, (x, y), which is the coordinate of destination.
#    closed_set: the set of Node's objects
#    selective_path: the list of tuples which are coordinates
#        ex. if our destination is (3, 3)
#        the output is = [(1, 1), (1, 2), (1, 3), (2,3)]
# we will move from (1, 1) to (3, 3) according to (1, 1) -> (1, 2) -> (1,
# 3) -> (2,3)
def backforward(end_node, closed_set, selective_path):

    # looking for all nodes in the closed_set
    for i in closed_set:
        # if once finding end_point is a child of the node, then this node will
        # be put in the front of selective_path.
        if end_node in i.children:
            selective_path.insert(0, i)

            # look backforward for the child of this node.
            return backforward(i, closed_set, selective_path)
    # if finding the beginning, it will return selective_path
    return selective_path

# Make us more closed to the next position, make the same direction
# This movment is for eating food


# Make us more closed to the next position, make the same direction
# This movment is for eating food
def moving(pcman, destination):

    if pcman[0] > destination[0]:
        return 'West'

    if pcman[0] < destination[0]:
        return 'East'

    if pcman[1] > destination[1]:
        return 'South'

    if pcman[1] < destination[1]:
        return 'North'

# Make us far away to ghosts, make the opposite direction
# This movment is for escaping from ghosts


def escape(xy_pacman, xy_ghost):
    tmp = []
    if (xy_pacman[0] - xy_ghost[0]) >= 0:
        tmp.append('East')
    if (xy_pacman[0] - xy_ghost[0]) <= 0:
        tmp.append('West')
    if (xy_pacman[1] - xy_ghost[1]) >= 0:
        tmp.append('North')
    if (xy_pacman[1] - xy_ghost[1]) <= 0:
        tmp.append('South')

    if len(tmp) == 0:
        tmp.append('Stop')

    return(set(tmp))


class TraceAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):

        name = "Pacman"

        # open a file for output

        # the list of food locations pcman has seen before
        self.seen_food_locations = []
        # whether pcman should find the new target to eat
        self.get_target = True
        # this is the tuple we use to store our target's coordinate
        self.target_coordinate = (0, 0)
        # this is the list of tuples storing coordinates, which pcman will
        # follow this path to move
        self.selective_path = []

        # ths is the coordinate of the last target
        self.last_target = []

    # This is what gets run in between multiple games
    def final(self, state):

        name = "Pacman"
        self.seen_food_locations = []
        self.get_target = True
        self.target_coordinate = (0, 0)
        self.selective_path = []
        self.last_target = []

        # self.outfile.close()
    # For now I just move randomly

    def getAction(self, state):

        # get legal action
        legal = api.legalActions(state)

        # get food locations
        food_locations = api.food(state)
        # get capsule location
        capsules_locations = api.capsules(state)
        # make food and capsule together
        food_locations.extend(capsules_locations)

        # put new food of capsule locations pcman see in this states
        self.seen_food_locations.extend(
            set(food_locations) - set(self.seen_food_locations))

        # get pcman location
        get_mylocation = api.whereAmI(state)
        # get ghost locations
        #detect_ghosts = api.ghosts(state)
        detect_ghosts = []

        feature_list = api.getFeatureVector(state)
        feature_list = feature_list[8:]
        # if len(detect_ghosts) > 0:
        if sum(feature_list) > 0:

            detect_ghosts = api.ghosts(state)
            detect_ghosts = list(filter(lambda x: hv_distance(
                x, get_mylocation) <= 2, detect_ghosts))

        # get wall locations
        wall_info = api.walls(state)

        # if pcman eats food at this state
        if get_mylocation in self.seen_food_locations:
            # remove this food location from the list storing the food
            # locations pcman has seen before
            self.seen_food_locations.remove(get_mylocation)
            # setting get_target as True means pcman should find new target for
            # the following steps.
            self.get_target = True

        # if pcman arrives at its target, we reset get_target to be True
        if get_mylocation == self.target_coordinate:
            self.get_target = True

        try:
            # to find the new target for the following steps
            if self.get_target:
                # pcman will choose the food which is the nearest to it.
                tmp = dict([(i, hv_distance(i, get_mylocation))
                            for i in self.seen_food_locations])
                if len(tmp) > 0:
                    self.target_coordinate = min(
                        tmp.keys(), key=lambda k: tmp[k])

                # if there is no food in seen_food_locations, we will randomly
                # chooce one of corner to be the new target
                else:
                    # we adjust corners' locations from api
                    get_corners = api.corners(state)
                    modify = [(min(i), max(i)) for i in zip(*get_corners)]
                    modify = [(i[0] + 1 if i[0] == modify[0][0] else i[0] - 1, i[1] +
                               1 if i[1] == modify[1][0] else i[1] - 1) for i in get_corners]
                    # when pcman have been all of the corners, reset again
                    # make sure pcman can go to 4 corners in 4 decisions
                    if len(self.last_target) == len(get_corners):
                        self.last_target = []

                    self.target_coordinate = random.choice(
                        [i for i in modify if i not in self.last_target])
                    self.last_target.append(self.target_coordinate)

                # We are going to use A* algorithm for finding the optimal route in order to eat its next target
                # set current location as the object of Node
                current_node = Node(get_mylocation)
                # initialise the open set for A* algorithm
                open_set = set([current_node])
                # initialise the closed set for A* algorithm
                closed_set = set([])
                # execute A* algorithm to find the closed set
                closed_set = A_Star(current_node, self.target_coordinate,
                                    wall_info, open_set, closed_set)

                # reconstruct the closed set to be the optimal route for the
                # next target and store this route to the list selective_path
                target_node = random.choice(
                    [i for i in closed_set if i.node == self.target_coordinate])

                self.selective_path = [
                    i.node for i in backforward(target_node, closed_set, [])]

                # we should add the coordinate of the new target at the end of
                # the list selective_path and remove the current locations
                # which is at the beginning of the list selective_path
                self.selective_path.append(self.target_coordinate)

                del self.selective_path[0]

                # pcman has decided the next target and found this optimal route for it
                # set get_target as False
                # becasue pcman will follow this route until it arrive the
                # target, get_target should be set as False
                self.get_target = False

            # For food, pcman will move from the current location to the next
            # location where is stored at the beginning of the list,
            # selective_path.

            action = moving(get_mylocation, self.selective_path[0])

            del self.selective_path[0]
        except:
            action = random.choice([i for i in legal if i != 'Stop'])
            self.get_target = True

        # if pcman meets ghost, it will chooce the direction which make it far
        # away from ghost
        if len(detect_ghosts) > 0:
            escape_direction = []
            # looking at all of the ghosts pcman observe
            for axis_ghosts in detect_ghosts:
                # for each ghost, pcman will find some directions to escape
                # from it
                try:
                    escape_direction.append(
                        escape(get_mylocation, axis_ghosts))
                except:
                    escape_direction.append({'Stop'})
            # find the direction which can escape all of the ghosts
            escape_direction = set.intersection(*escape_direction)
            # make sure this is a legal movement
            escape_direction = escape_direction.intersection(set(legal))
            # if there is no action for escaping from ghosts, it will stop for
            # the next step
            if len(escape_direction) == 0:
                escape_direction = set(['Stop'])

            # if there is no action which can make pcman more closed to food
            # and far away from ghosts
            if action not in escape_direction:
                # it will select one of the action for escaping randomly.
                action = random.choice(list(escape_direction))
                # At this moment, it leave from the optimal route, it will need to find a new target and the optimal route for it at the next state.
                # Hence, get_target should be set as True.
                self.get_target = True

        if os.path.exists('moves2.txt'):

            outfile = open('moves2.txt', 'a')
        else:
            outfile = open('moves2.txt', 'w')

        if action != 'Stop':
            outfile.write(api.getFeaturesAsString(state))
            if action == 'North':
                outfile.write("0\n")
            elif action == 'East':
                outfile.write("1\n")
            elif action == 'South':
                outfile.write("2\n")
            elif action == 'West':
                outfile.write("3\n")
        outfile.close()

        return api.makeMove(action, legal)

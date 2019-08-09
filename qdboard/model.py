import numpy as np
import math
import numpy


class NoPathError(Exception):
    pass


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end, max_iterations=200):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    adjacent = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Loop until you find the end
    iterations = 0
    while len(open_list) > 0:

        iterations += 1

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in adjacent: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

        if iterations >= max_iterations:
            raise NoPathError("No path ")


class Dimension:

    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value

    def to_json(self):
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value
        }


class Archive:

    def __init__(self, dimensions, cells, solutions):
        self.cells = cells
        self.dimensions = dimensions
        self.solutions = solutions
        self.fitnesses = [solution.fitness for solution in solutions]

    def to_json(self):
        return {
            'cells': [cell.to_json() for cell in list(self.cells.values())],
            #'solutions': [solution.to_json() for solution in self.solutions],
            'dimensions': [dim.to_json() for dim in self.dimensions],
            'fitness_std': np.std(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_mean': np.mean(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_min': np.min(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_max': np.max(self.fitnesses) if len(self.fitnesses) > 0 else None
        }


class Cell:

    def __init__(self, points, solutions):
        self.points = points
        self.solutions = solutions
        self.fitnesses = [solution.fitness for solution in solutions]

    def add_solution(self, solution):
        self.solutions.append(solution)
        self.fitnesses.append(solution.fitness)

    def to_json(self):
        return {
            'points': self.points,
            'solutions': [solution.to_json() for solution in self.solutions],
            'fitness_mean': np.mean(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_std': np.std(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_min': np.min(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_max': np.max(self.fitnesses) if len(self.fitnesses) > 0 else None
        }


class Solution:

    def __init__(self, genotype, behavior, fitness):
        self.genotype = genotype
        self.behavior = behavior
        self.fitness = fitness

    def to_json(self):
        return {
            'genotype': self.genotype,
            'behavior': self.behavior,
            'fitness': self.fitness
        }


class QDAlgorithm:

    def __init__(self, run_id, config, b_dimensions, problem):
        self.run_id = run_id
        self.config = config
        self.b_dimensions = b_dimensions
        self.b_dims = len(self.b_dimensions)
        self.problem = problem
        self.b_mins = [dimension.min_value for dimension in self.b_dimensions]
        self.b_maxs = [dimension.max_value for dimension in self.b_dimensions]

    def start(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def stop(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def get_archive(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def is_done(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def to_json(self):
        return {
            'run_id': self.run_id,
            'b_dimensions': [dim.to_json() for dim in self.b_dimensions],
            'problem': self.problem.to_json(),
            'is_done': self.is_done(),  # Fix this
            'b_mins': self.b_mins,
            'b_maxs': self.b_maxs
        }


class Problem:

    def __init__(self, name, x_dims, b_dims, min_fit, max_fit, x_min=0, x_max=1, continuous=True, blocks=None):
        self.name = name
        self.x_dims = x_dims
        self.b_dims = b_dims
        self.continuous = continuous
        self.blocks = blocks
        self.x_min = x_min
        self.x_max = x_max
        self.min_fit = min_fit
        self.max_fit = max_fit

    def evaluate(self, genotype):
        raise NotImplementedError("Must be overridden by sub-class")

    def to_json(self):
        return {
            'name': self.name,
            'x_dims': self.x_dims,
            'b_dims': self.b_dims,
            'continous': self.continuous,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'min_fit': self.min_fit,
            'max_fit': self.max_fit
        }


class Rastrigin(Problem):

    def __init__(self, x_dims, b_dims, min_fit=-100, max_fit=0):
        super().__init__(f"Rastrigin-{x_dims}D-{b_dims}D", x_dims, b_dims, min_fit, max_fit)

    def evaluate(self, genotype):
        fitness = self.__rastrigin(genotype)
        behavior = genotype[:2]
        return Solution(genotype, behavior, fitness)

    def __rastrigin(self, xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f


class Zelda(Problem):

    def __init__(self, width=13, height=9, b_dims=2, min_fit=-100, max_fit=0, max_danger=24):
        super().__init__(f"Zelda-{width}-{height}D-{b_dims}D", width*height, b_dims, min_fit, max_fit, continuous=False, blocks=['.','w','1','2','3','g','A','+'])
        self.width = width
        self.height = height
        self.max_danger = max_danger

    def evaluate(self, genotype):

        free = self.__count(genotype, ['.'])
        danger = self.__danger(genotype)
        fitness = self.__fitness(genotype)

        #if fitness > 0:
        #    self.__print(genotype)

        behavior = [
            min(self.max_danger, danger),
            free
        ]

        return Solution(genotype, behavior, fitness)

    def __danger(self, genotype):
        enemies1 = self.__count(genotype, ['1'])
        enemies2 = self.__count(genotype, ['2'])
        enemies3 = self.__count(genotype, ['3'])
        danger = enemies1 * 2 + enemies2 * 3 + enemies3 * 4
        return danger

    def __fitness(self, genotype):
        padding = self.__count(genotype, ['w'], padding=True)
        keys = self.__count(genotype, ['+'])
        doors = self.__count(genotype, ['g'])
        agents = self.__count(genotype, ['A'])
        danger = self.__danger(genotype)

        path_to_key_bonus = -20
        path_to_door_bonus = -20

        if agents == 1:

            maze = self.__get_maze(genotype)

            agent_location = self.__get_location(genotype, 'A')

            if keys == 1:
                key_location = self.__get_location(genotype, '+')
                try:
                    from_agent_to_key = astar(maze, agent_location, key_location)
                    if from_agent_to_key is not None:
                        path_to_key_bonus = len(from_agent_to_key)
                except NoPathError as e:
                    pass

                if doors == 1:
                    door_location = self.__get_location(genotype, 'g')
                    try:
                        from_key_to_door = astar(maze, key_location, door_location)
                        if from_key_to_door is not None:
                            path_to_door_bonus = len(from_key_to_door)
                    except NoPathError as e:
                        pass

        danger_bonus = 0
        #if danger > self.max_danger:
        #    danger_bonus = self.max_danger - danger

        return -abs(agents - 1)*5 - abs(keys - 1)*5 - abs(doors - 1)*5 - (self.width*2 + (self.height-2)*2) + padding + danger_bonus - 20*2 + path_to_key_bonus + path_to_door_bonus

    def __get_location(self, genotype, block):
        for y in range(self.height):
            for x in range(self.width):
                i = y*self.width + x
                if genotype[i] == block:
                    return (x, y)
        raise Exception("No block found of type " + block)

    def __get_maze(self, genotype):
        maze = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                i = y * self.width + x
                if genotype[i] == 'w':
                    maze[y][x] = 1
        return maze

    def __print(self, genotype):
        print("--------------")
        padding = self.__count(genotype, ['w'], padding=True)
        keys = self.__count(genotype, ['+'])
        doors = self.__count(genotype, ['g'])
        agents = self.__count(genotype, ['A'])
        # enemies = self.__count(genotype, ['1', '2', '3'])
        enemies1 = self.__count(genotype, ['1'])
        enemies2 = self.__count(genotype, ['2'])
        enemies3 = self.__count(genotype, ['3'])
        # blocks = self.__count(genotype, ['w'])
        free = self.__count(genotype, ['.'])
        fitness = self.__fitness(genotype)
        print(f"FITNESS: {fitness}")
        print(f"PADDING: {padding}")
        print(f"KEYS: {keys}")
        print(f"DOORS: {doors}")
        print(f"AGENTS: {agents}")
        print(f"ENEMIES1: {enemies1}")
        print(f"ENEMIES2: {enemies2}")
        print(f"ENEMIES3: {enemies3}")
        print(f"FREE: {free}")
        print(f"LEVEL:")

        for y in range(self.height):
            line = ""
            for x in range(self.width):
                i = y*self.width + x
                line += genotype[i]
            print(line)
        print("--------------")
        fitness = self.__fitness(genotype)

    def __count(self, genotype, blocks, padding=False):
        c = 0
        for y in range(self.height):
            for x in range(self.width):
                i = y*self.width + x
                if not padding or (x == 0 or y == 0 or x == self.width-1 or y == self.height-1):
                    if genotype[i] in blocks:
                        c += 1

        return c




def main():

    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)

    path = astar(maze, start, end)
    print(path)


if __name__ == '__main__':
    main()
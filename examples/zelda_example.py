#!/usr/bin/env python3

import os
import json
import numpy as np
import uuid
import qdboard
from qdboard.algos.map_elites import MapElites
from qdboard.api import add_run
import qdboard.server as server
from qdboard.model import Problem, Dimension, Solution, ImgVisualizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image


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


# Define problem to optimize
class Zelda(Problem):

    def __init__(self, width=13, height=9, b_dims=2, min_fit=-100, max_fit=0, max_danger=24, vary=None):
        super().__init__(f"Zelda-{width}-{height}D-{b_dims}D", width*height, b_dims, min_fit, max_fit, continuous=False, blocks=['.','w','1','2','3','g','A','+'])
        self.width = width
        self.height = height
        self.max_danger = max_danger
        self.vary = vary

    def evaluate(self, genotype):

        free = self.__count(genotype, ['.'])
        danger = self.__danger(genotype)
        if self.vary is None:
            fitness = self.__fitness(genotype)
        else:
            fitness = self.__fitness_similarity(genotype)

        #if fitness > 0:
        #    self.__print(genotype)

        behavior = [
            min(self.max_danger, danger),
            free
        ]

        return Solution(str(uuid.uuid1()), genotype, behavior, phenotype=genotype, fitness=fitness, img=None)

    def __fitness_similarity(self, genotype):
        diff = [(1 if genotype[i] == self.vary[i] else 0) for i in range(len(genotype))]
        return np.sum(diff) / (self.width * self.height)

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
                    from_agent_to_key = self.__astar(maze, agent_location, key_location)
                    if from_agent_to_key is not None:
                        path_to_key_bonus = len(from_agent_to_key)
                except NoPathError as e:
                    pass

                if doors == 1:
                    door_location = self.__get_location(genotype, 'g')
                    try:
                        from_key_to_door = self.__astar(maze, key_location, door_location)
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

    def __astar(self, maze, start, end, max_iterations=200):
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
                return path[::-1]  # Return reversed path

            # Generate children
            children = []
            for new_position in adjacent:  # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
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
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                            (child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

            if iterations >= max_iterations:
                raise NoPathError("No path ")


class SpriteSheetReader:

    def __init__(self, imageName, tileSize):
        self.spritesheet = Image.open(imageName)
        self.tileSize = tileSize
        self.margin = 1

    def getTile(self, tileX, tileY):
        posX = (self.tileSize * tileX) + (self.margin * (tileX + 1))
        posY = (self.tileSize * tileY) + (self.margin * (tileY + 1))
        box = (posX, posY, posX + self.tileSize, posY + self.tileSize)
        return self.spritesheet.crop(box)


class SpriteSheetWriter:

    def __init__(self, tileSize, width, height):
        self.tileSize = tileSize
        self.width = width
        self.height = height
        self.spritesheet = Image.new("RGBA", (self.width*tileSize, self.height*tileSize), (0, 0, 0, 0))
        self.tileX = 0
        self.tileY = 0
        self.margin = 0

    def getCurPos(self):
        self.posX = (self.tileSize * self.tileX) + (self.margin * (self.tileX + 1))
        self.posY = (self.tileSize * self.tileY) + (self.margin * (self.tileY + 1))
        if (self.posX + self.tileSize > self.width*self.tileSize):
            self.tileX = 0
            self.tileY = self.tileY + 1
            self.getCurPos()
        if (self.posY + self.tileSize > self.height*self.tileSize):
            raise Exception('Image does not fit within spritesheet!')

    def addImage(self, image):
        self.getCurPos()
        destBox = (self.posX, self.posY, self.posX + image.size[0], self.posY + image.size[1])
        self.spritesheet.paste(image, destBox)
        self.tileX = self.tileX + 1

    def save(self, filename):
        self.spritesheet.save(filename)

    def close(self):
        self.spritesheet.close()


class ZeldaImgVisualizer(ImgVisualizer):

    def __init__(self, path, width=13, height=9):
        super().__init__('ZeldaVisualizer', path)
        self.width = width
        self.height = height
        self.tiles = {
            '.': Image.open('/Users/njustesen/git/qdboard/examples/img/f.png'),
            '1': Image.open('/Users/njustesen/git/qdboard/examples/img/1.png'),
            '2': Image.open('/Users/njustesen/git/qdboard/examples/img/2.png'),
            '3': Image.open('/Users/njustesen/git/qdboard/examples/img/3.png'),
            'A': Image.open('/Users/njustesen/git/qdboard/examples/img/A.png'),
            'f': Image.open('/Users/njustesen/git/qdboard/examples/img/f.png'),
            'g': Image.open('/Users/njustesen/git/qdboard/examples/img/g.png'),
            'w': Image.open('/Users/njustesen/git/qdboard/examples/img/w.png'),
            '+': Image.open('/Users/njustesen/git/qdboard/examples/img/+.png')
        }

    def save_visualization(self, solution):
        fname = os.path.join(self.path, f'{solution.solution_id}.png')
        if not os.path.isfile(fname):
            self.__visualize_level(solution.phenotype, title=str(solution.solution_id))

    def get_rel_path(self, solution):
        title = str(solution.solution_id)
        path = os.path.join('static', self.path.split('/static/')[1])
        return os.path.join(path, f'{title}.png')

    def __visualize_level(self, level, title=None):

        writer = SpriteSheetWriter(24, width=self.width, height=self.height)

        for i in range(len(level)):
            writer.addImage(self.tiles[level[i]])

        writer.save(os.path.join(self.path, f'{title}.png'))
        writer.close()


class FileHandler:

    def load_level(self, path):
        level = []
        with open(path) as f:
            for line in f:
                for c in line:
                    if c != '\n':
                        level.append(c)
        return level


def get_data_path(rel_path):
    root_dir = qdboard.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, rel_path)
    return os.path.abspath(os.path.realpath(filename))


config_zelda = {
    "cvt_samples": 25000,
    "batch_size": 100,
    "random_init": 50,
    "random_init_batch": 50,
    "sigma_iso": 0.01,
    "sigma_line": 0.2,
    "dump_period": 10,
    "parallel": True,
    "cvt_use_cache": True,
    "archive_path": get_data_path("map-elites/runs/"),
    "centroids_path": get_data_path("map-elites/centroids/"),
    "num_niches": 1000,
    "num_gens": 100000,
    "discrete_muts": 20,
    "discrete_mut_prob": 0.2,
    "block_probs": [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}

width = 13
height = 9
spaces = ((width*height)-(width*2+height*2))
b_dimensions_zelda = [
    Dimension("Danger", max_value=spaces, min_value=0),
    Dimension("Openness", max_value=spaces, min_value=0),
]
run_id = str(uuid.uuid1())

human_level = FileHandler().load_level(get_data_path("data/zelda/zelda_lvl0.txt"))
problem_zelda = Zelda(width, height, len(b_dimensions_zelda), min_fit=-150, max_fit=0, max_danger=spaces)
problem_zelda_vary = Zelda(width, height, len(b_dimensions_zelda), min_fit=0, max_fit=1, max_danger=spaces, vary=human_level)
human_level_solution = problem_zelda_vary.evaluate(human_level)

'''
dim_lens = [b_dimensions_zelda[i].max_value - b_dimensions_zelda[i].min_value for i in range(len(b_dimensions_zelda))]
b_dimensions_zelda_vary = [
    Dimension("Danger", max_value=human_level_solution.behavior[0] + dim_lens[0]/2, min_value=human_level_solution.behavior[0] - dim_lens[0]/2),
    Dimension("Openness", max_value=human_level_solution.behavior[1] + dim_lens[1]/2, min_value=human_level_solution.behavior[1] - dim_lens[1]/2),
]
'''

visualizer = ZeldaImgVisualizer(get_data_path(os.path.join('static/img/zelda/', run_id)))
algo_zelda = MapElites(run_id, config_zelda, b_dimensions=b_dimensions_zelda, problem=problem_zelda, img_visualizer=visualizer)
# algo_zelda = MapElites(run_id, config_zelda, b_dimensions=b_dimensions_zelda, problem=problem_zelda_vary, img_visualizer=visualizer)
add_run(algo_zelda)
algo_zelda.start()

# Run server
server.start_server(debug=True, use_reloader=False, port=5000)

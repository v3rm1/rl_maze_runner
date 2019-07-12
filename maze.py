import random

import numpy as np
from matplotlib import pyplot as plt


class Maze:
    """
    Class defining the maze, starting and end points.
    Maze is generated by depth-first search maze generation algorithm.
    """
    def __init__(self):
        self.maze = None
        self.start = None
        self.end = None
        self.path = None
        self.sel_block = None
        self.block_size = 40
        self.WALL_COLOR = [0,0,0]
        self.BACKGROUND_COLOR_INT = 255
        self.BACKGROUND_COLOR = [self.BACKGROUND_COLOR_INT for i in range(0, 3, 1)]
        self.START_COLOR = [255, 0, 0]
        self.GOAL_COLOR = [0, 255, 0]
        self.PATH_COLOR = [0, 0, 255]
        self.SELECTION_COLOR = [232, 244, 66]


    def create_maze(self, height, width):
        self.maze = [[None for i in range(0, width, 1)] for j in range(0, height, 1)]
        curr_pt = (0, 0)
        self.maze[0][0] = curr_pt
        maze_stack = [curr_pt]
        unv_spaces = (height * width) - 1
        while(unv_spaces > 1):
            curr_unv_neighbors = self._get_unv_neighbors(curr_pt)
            if len(curr_unv_neighbors) == 0:
                curr_pt = maze_stack.pop()
            else:
                next_pt = random.choice(curr_unv_neighbors)
                self.maze[next_pt[0]][next_pt[1]] = curr_pt
                maze_stack.append(curr_pt)
                curr_pt = next_pt
                unv_spaces -= 1

    def set_start(self):
        start_x = random.randint(0, len(self.maze[0]) // 2)
        start_y = random.randint(0, len(self.maze) // 2)
        self.start = (start_y, start_x)
    
    def get_start(self):
        return self.start
    
    def set_end(self):
        end_x = random.randint(len(self.maze[0]) // 2, len(self.maze[0]))
        end_y = random.randint(len(self.maze) // 2, len(self.maze))
        self.end = (end_y, end_x)
    
    def get_end(self):
        return self.end

    def get_maze(self):
        return self.maze

    def set_path(self, path):
        self.path = path

    def set_sel_block(self, sel_block_coords):
        self.sel_block = sel_block_coords

    def clear_selected_block(self):
        self.sel_block = None

    def get_block_size(self):
        return self.block_size

    def gen_image(self):
        self.set_start()
        self.set_end()
        if self.maze is None:
            raise ValueError("Maze is not initialized")

        image = self._initialize_image()

        if self.path is not None:
            image = self._draw_path(image, self.path)
        if self.sel_block is not None:
            image = self._draw_sel_block(image, self.sel_block)
        if self.start is not None:
            image = self._draw_start(image, self.start)
        if self.end is not None:
            image = self._draw_end(image, self.end)

        image = self._draw_walls(image)
        for y in range(0, len(self.maze), 1):
            for x in range(0, len(self.maze[0]), 1):
                if (x != 0 or y != 0):
                    if self.maze[y][x] == None:
                        self.maze[y][x] = (0, 0)
                    image = self._remove_walls(image, (y, x), self.maze[y][x])
        return image
    
    def _get_unv_neighbors(self, pt):
        unv_neighbors = []
        for offsets in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            y_ind = pt[0] + offsets[0]
            x_ind = pt[1] + offsets[1]

            if y_ind >= 0 and y_ind < len(self.maze):
                if x_ind >= 0 and x_ind < len(self.maze[0]):
                    if self.maze[y_ind][x_ind] is None:
                        unv_neighbors.append((y_ind, x_ind))
        return unv_neighbors

    def _initialize_image(self):
        image = np.full((len(self.maze) * self.block_size, len(self.maze[0]) * self.block_size, 3), self.BACKGROUND_COLOR_INT, dtype=np.uint8)
        return image


    def _draw_walls(self, image):
        for y in range(0, image.shape[0], 1):
            for x in range(0, image.shape[1], 1):
                is_wall = False
                if y % self.block_size == 0 or y % self.block_size == self.block_size - 1:
                    is_wall = True
                if x % self.block_size == 0 or x % self.block_size == self.block_size - 1:
                    is_wall = True
                if is_wall:
                    image[y][x] = self.WALL_COLOR
        return image

    def _remove_walls(self, image, pt1, pt2):
        print("Point 1: {0} \nPoint 2: {1}".format(pt1, pt2))
        if pt1[0] == pt2[0] and pt1[1] == pt2[1]:
            raise ValueError('Duplicate points: {0} and {1}'.format(pt1, pt2))
        
        if (pt1[0] == pt2[0]):
            if (pt1[1] > pt2[1]):
                image = self.__remove_left_wall(image, pt1)
                image = self.__remove_right_wall(image, pt2)
            elif (pt1[1] < pt2[1]):
                image = self.__remove_right_wall(image, pt1)
                image = self.__remove_left_wall(image, pt2)
                

        elif (pt1[1] == pt2[1]):
            if (pt1[0] > pt2[0]):
                image = self.__remove_top_wall(image, pt1)
                image = self.__remove_bottom_wall(image, pt2)
            elif (pt1[0] < pt2[0]):
                image = self.__remove_bottom_wall(image, pt1)
                image = self.__remove_top_wall(image, pt2)
        else:
            raise ValueError('points {0} and {1} cannot be connected.'.format(pt1, pt2))
        return image


    def __remove_top_wall(self, image, pt):
        y_px = (pt[0]) * self.block_size
        x_init = (pt[1] * self.block_size) + 1
        x_fin = ( (pt[1]+1) * self.block_size ) - 1

        color = self.BACKGROUND_COLOR
        if self.path is not None and pt in self.path:
            color = self.PATH_COLOR

        while x_init < x_fin:
            image[y_px][x_init] = color
            x_init += 1
        return image


    def __remove_bottom_wall(self, image, pt):
        y_px = ((pt[0]+1) * self.block_size) - 1
        x_init = (pt[1] * self.block_size) + 1
        x_fin = ((pt[1]+1) * self.block_size) - 1

        color = self.BACKGROUND_COLOR
        if self.path is not None and pt in self.path:
            color = self.PATH_COLOR

        while x_init < x_fin:
            image[y_px][x_init] = color
            x_init += 1
        return image


    def __remove_left_wall(self, image, pt):
        x_px = (pt[1]) * self.block_size
        y_init = (pt[0] * self.block_size) + 1
        y_fin = ( (pt[0]+1) * self.block_size ) - 1
        
        color = self.BACKGROUND_COLOR
        if self.path is not None and pt in self.path:
            color = self.PATH_COLOR

        while y_init < y_fin:
            image[y_init][x_px] = color
            y_init += 1
        return image


    def __remove_right_wall(self, image, pt):
        x_px = ((pt[1] + 1) * self.block_size) - 1
        y_init = (pt[0] * self.block_size) + 1
        y_fin = ((pt[0]+1) * self.block_size) - 1

        color = self.BACKGROUND_COLOR
        if self.path is not None and pt in self.path:
            color = self.PATH_COLOR

        while y_init < y_fin:
            image[y_init][x_px] = color
            y_init += 1
        return image

    def _draw_start(self, image, pt):
        return self.__draw_square(image, pt, self.START_COLOR)
    
    def _draw_end(self, image, pt):
        return self.__draw_square(image, pt, self.GOAL_COLOR)

    def __draw_square(self, image, pt, color):
        center_y = (pt[0] * self.block_size) + (self.block_size // 2)
        center_x = (pt[1] * self.block_size) + (self.block_size // 2)

        min_x = center_x - 3
        min_y = center_y - 3
        max_x = center_x + 3
        max_y = center_y + 3
        return self.__draw_rectangle(image, min_x, min_y, max_x, max_y, color)


    def _draw_path(self, image, path):
        for path_node in path:
            min_x = path_node[1] * self.block_size
            min_y = path_node[0] * self.block_size
            max_x = (path_node[1] + 1) * self.block_size
            max_y = (path_node[1] + 1) * self.block_size
            image = self.__draw_rectangle(image, min_x, min_y, max_x, max_y, self.PATH_COLOR)
        return image

    def _draw_sel_block(self, image, sel_block):
        min_x = sel_block[1] * self.block_size
        min_y = sel_block[0] * self.block_size
        max_x = (sel_block[1] + 1) * self.block_size
        max_y = (sel_block[1] + 1) * self.block_size
        image = self.__draw_rectangle(image, min_x, min_y, max_x, max_y, self.SELECTION_COLOR)
        return image

    def __draw_rectangle(self, image, min_x, min_y, max_x, max_y, color):
        for y in range(min_y, max_y, 1):
            for x in range(min_x, max_x, 1):
                image[y][x] = color
        return image



if __name__ == "__main__":
    mz = Maze()
    mz.create_maze(6, 6)
    mz_img = mz.gen_image()
    plt.imshow(mz_img)
    plt.show()
    plt.imsave("mz_6_6.jpg", mz_img)
import numpy as np
import random
import sys
import maze

# Defining the Q-Learning agent
class Agent:

    def __init__(self):
        self.R = None
        self.Q = None

        self.col_num = None
        self.end_state = None
        self.num_states = None
        self.next_states = {}
        self.trained = False

    def is_trained(self):
        """
        Tracking training state
        Arguments:
            Instance of class Agent
        Returns:
            self.trained: bool
        """
        return self.trained
    
    def initialize_maze(self, maze):
        """
        Initializing maze components and values
        Arguments:
        self: Instance of class Agent
        maze: Instance of class Maze
        Returns:
        """
        if maze is None:
            return None
            
        maze_fields = maze.get_maze()
        end_point = maze.get_end_point()

        if end_point is None:
            return
        self.num_states = len(maze_fields) * len(maze_fields[0])
        self.col_num = len(maze_fields[0])

        self.R = np.full((self.num_states, self.num_states), -1, dtype=np.float64)
        for y in range(0, len(maze_fields), 1):
            for x in range(0, len(maze_fields[0]), 1):
                pt_1 = (y, x)
                pt_2 = maze_fields[y][x]

                if (pt_1[0] != pt_2[0] or pt_1[1] != pt_2[1]):
                    state_1 = self.__maze_dims_to_state(pt_1[0], pt_1[1])
                    state_2 = self.__maze_dims_to_state(pt_2[0], pt_2[1])
                    
                    if state_1 not in self.next_states:
                        self.next_states[state_1] = []
                    self.next_states[state_1].append(state_2)
                    if state_2 not in self.next_states:
                        self.next_states[state_2] = []
                    self.next_states[state_2].append(state_1)

                    self.R[state_1][state_2] = 0
                    self.R[state_2][state_1] = 0

        self.end_state = self.__maze_dims_to_state(end_point[0], end_point[1])
        for i in range(0, self.num_states, 1):
            if self.R[i][self.end_state] != -1:
                self.R[i][self.end_state] = 1.0
        self.R[self.end_state][self.end_state] = 1.0

        self.Q = np.full((self.num_states, self.num_states), 0.0)
        self.trained = False
    
    def train_agent(self, gamma, min_update_per_epoch):
        print("Training Q-Learning agent.")
        epoch_iteration = 0
        while True:
            prev_q = np.copy(self.Q)

            for i in range(0, 10, 1):
                state_curr = random.randint(0, self.num_states-1)
                while(state_curr != self.end_state):
                    poss_next_states = self.next_states[state_curr]
                    state_t_1 = random.choice(poss_next_states)
                    max_q_next_state = -1
                    states_t_2 = self.next_states[state_t_1]

                    for state_t_2 in states_t_2:
                        max_q_next_state = max([max_q_next_state, self.Q[state_t_1][state_t_2]])
                    slef.Q[state_curr][state_t_1] = self.R[state_curr][state_t_1] + (gamma * max_q_next_state)

                    state_curr = state_t_1
            self.Q = self.Q / np.max(self.Q)

            diff = np.sum(np.abs(self.Q - prev_q))
            print("Epoch: {}\nPrev_Q: {1}\tUpdated_Q:{2}\tDifference:{3}".format(epoch_iteration, self.Q, prev_q, diff))

            if(diff < min_update_per_epoch):
                break
            epoch_iteration += 1
        self.trained = True

    def solve(self, start_state):
        if not self.trained:
            return []
        path = [start_state]
        state_curr = self.__maze_dims_to_state(start_state[0], start_state[1])

        while(state_curr != self.end_state and len(path) < self.num_states):
            poss_next_states = self.next_states[state_curr]
            best_state = poss_next_states[0]
            best_state_reward = self.Q[state_curr][best_state]

            for i in range(1, len(poss_next_states), 1):
                pot_next_state = poss_next_states[i]
                if(self.Q[state_curr][pot_next_state] > best_state_reward):
                    best_state = pot_next_state
                    best_state_reward = self.Q[state_curr][pot_next_state]
            state_curr = best_state
            path.append(self.__state_to_maze_dims(state_curr))
        return path

    def __maze_dims_to_state(self, y, x):
        return (y*self.col_num) + x

    def __state_to_maze_dims(self, state):
        y = int(state // self.col_num)
        x = state % self.col_num
        return (y, x)


import numpy as np
from matplotlib import pyplot as plt
from queue import Queue
# This code uses another algorithm and NOT greedy best first search

def read_maze(filename: str):
    """
    Reads a txt file and returns the data as a array in the correct shape
    :param filename: Name of the file to be used
    :return: A Maze represented by a numpy array
    """
    data = np.genfromtxt(filename, delimiter="", dtype=str)
    width, height = len(data[0]), len(data)
    data = [int(i) for num in data for i in str(num)]
    data = np.array(data)
    maze = data.reshape(height, width)
    return maze


def get_neighbors(maze, row: int, col: int, connectivity: int):
    dirs = None
    neighbors = []

    if connectivity == 4:
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    elif connectivity == 8:
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (1, 1), (-1, 1)]

    for drow, dcol in dirs:
        new_row, new_col = row + drow, col + dcol
        if 0 <= new_row < len(maze) and 0 <= new_col < len(maze[0]) and maze[new_row][new_col] in [0, 3]:
            neighbors.append((new_row, new_col))
    return neighbors


def bfs(maze, start_row: int, start_col: int, connectivity: int):
    # TODO: FÖR FAN FEL ALGORYTM 
    """
    Perform a Breadth First Search (BFS) algorithm on a maze to find the shortest path from the start
    cell to the goal cell.
    :param maze:
    :param start_row:
    :param start_col:
    :param connectivity:
    :return:
    """

    q = Queue()
    q.put((start_row, start_col))
    visited = set()
    path = {}
    visited.add((start_row, start_col))
    while not q.empty():
        row, col = q.get()
        if maze[row][col] == 3:
            # path found, backtrack to find path
            path_list = []
            while (row, col) in path:
                path_list.append((row, col))
                row, col = path[(row, col)]
            path_list.append((start_row, start_col))
            return path_list[::-1], visited
        for neighbor in get_neighbors(maze, row, col, connectivity):
            if neighbor not in visited:
                visited.add(neighbor)
                path[neighbor] = (row, col)
                q.put(neighbor)
    # no path found
    raise ValueError("No solution found!")


def starting_point(maze: np.array):
    """
    Searches for a "2" that indicates the starting point of a maze
    :param maze: numpy array of maze
    :return: start_row:
    :return: start_col
    """
    start_row, start_col = None, None
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 2:
                start_row, start_col = row, col
                return start_row, start_col

    if start_row is None or start_col is None:
        print("Error: maze does not have a start point")
        exit(1)


def plot(maze: np.array, path: list, visited):
    """
    :param maze: numpy array of maze
    :param path: list of tuples containing path coordinates from start to goal
    :return:
    """
    # Init the figure
    fig, ax = plt.subplots()
    # Go over every ppoint in the maze
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # Can't go
            if maze[i][j] == 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
            # Start point
            elif maze[i][j] == 2:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='deeppink'))
            # Goal
            elif maze[i][j] == 3:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))
            # Path
            elif (i, j) in path:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lime'))
            #
            else:
                if (i, j) in visited:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lightgreen'))
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white'))

    ax.set_xlim(0, len(maze[0]))
    ax.set_ylim(0, len(maze))
    ax.invert_yaxis() # Idk why the Y axis gets inverted
    plt.show()


def main():
    """
    Main function that calls for the necesarry functions in the correct order
    """
    big_maze = "maze_big.txt"
    small_maze = "maze_small.txt"
    maze = read_maze(big_maze)
    start_row, start_col = starting_point(maze)
    solution_4_connectivity, visited_4 = bfs(maze, start_row, start_col, 4)
    solution_8_connectivity, visited_8 = bfs(maze, start_row, start_col, 8)
    print("4-Connectivity Path:", solution_4_connectivity)
    print("8-Connectivity Path:", solution_8_connectivity)
    plot(maze, solution_4_connectivity, visited_4)
    plot(maze, solution_8_connectivity, visited_8)

if __name__ == "__main__":
    main()
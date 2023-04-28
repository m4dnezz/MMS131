import numpy as np
from matplotlib import pyplot as plt


############################################
# Made by Niclas Persson 2023/04/26
###########################################


def read_maze(filename: str):
    data = np.genfromtxt(filename, delimiter="", dtype=str)
    width, height = len(data[0]), len(data)
    data = [int(i) for num in data for i in str(num)]
    data = np.array(data)
    maze = data.reshape(height, width)
    return maze


def get_neighbors(maze, point, connectivity: int):
    dirs = None
    neighbors = []
    col = point[1]
    row = point[0]

    if connectivity == 4:
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    elif connectivity == 8:
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (1, 1), (-1, 1)]

    for drow, dcol in dirs:
        new_row, new_col = row + drow, col + dcol
        if 0 <= new_row < len(maze) and 0 <= new_col < len(maze[0]) and int(maze[new_row][new_col]) in [0, 3]:
            neighbors.append((new_row, new_col))
    return neighbors


def bfs(maze, start: tuple, goal: tuple, connectivity: int):
    def dist(node):
        return np.sqrt(np.abs((node[0] - goal[0])) ** 2 + np.abs((node[1] - goal[1]))**2) # Euclidean
        # return abs(node[0] - goal[0]) + abs(node[1] - goal[1]) # Manhattan

    # Initialize the frontier as a list containing the starting node and cost
    frontier = [(start, 0)]

    # Initialize the came_from dictionary with the starting node
    came_from = {start: None}

    while frontier:
        # Get the node with the lowest heuristic cost from the frontier
        current, cost = min(frontier, key=lambda x: x[1])
        # Remove the current node from the frontier
        frontier.remove((current, cost))

        # If we've reached the goal node, reconstruct the path and return it
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print(came_from)
            return path

        # Loop through the neighboring nodes of the current node
        for neighbor in get_neighbors(maze, current, connectivity):
            # Calculate the heuristic cost of the node
            heuristic_cost = dist(neighbor)
            # If the neighbor hasn't been visited before, add it to the frontier and came_from
            if neighbor not in came_from:
                frontier.append((neighbor, heuristic_cost))
                came_from[neighbor] = current

    # If we've exhausted the frontier and haven't found the goal, return None
    raise Exception("No solution found!")


def starting_point(maze: np.array):
    start_row, start_col, goal_row, goal_col = None, None, None, None
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 2:
                start_row, start_col = row, col
            elif maze[row][col] == 3:
                goal_row, goal_col = row, col

    if start_row is None or start_col is None:
        print("Error: maze does not have a start point")
        exit(1)
    else:
        return (start_row, start_col), (goal_row, goal_col)


def plot(maze: np.array, path: list):
    # Init the figure
    fig, ax = plt.subplots()
    path_label = False
    wall_label = False
    # Go over every ppoint in the maze
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # Can't go
            if maze[i][j] == 1:
                if not wall_label:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black', label="Wall"))
                    wall_label = True
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

            # Start point
            elif maze[i][j] == 2:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='blue', label="Start"))

            # Goal
            elif maze[i][j] == 3:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red', label="Goal"))

            # Path
            elif (i, j) in path:
                if not path_label:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lime', label="Path"))
                    path_label = True
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lime'))

            else:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white'))

    ax.set_xlim(0, len(maze[0]))
    ax.set_ylim(0, len(maze))
    ax.invert_yaxis()
    plt.legend()
    plt.show()


def main(connectivity: int, filename: str):
    if connectivity not in [4, 8]:
        raise ValueError("Connectivity must be 4 or 8")

    maze = read_maze(filename)
    start, goal = starting_point(maze)
    solution = bfs(maze, start, goal, connectivity)
    print("8-Connectivity Path:", solution)
    plot(maze, solution)


# Press Green arrow to run
# Files and connectivity is changed below and below only
if __name__ == "__main__":
    main(connectivity=8, filename="maze_big.txt")

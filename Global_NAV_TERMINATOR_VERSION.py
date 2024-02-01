import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

############################################################################################################################################

def check_matrix(matrix):
    
    nb_start = np.size(np.argwhere(matrix == 2),0)
    nb_goal = np.size(np.argwhere(matrix == 3),0)
    if nb_start == 1 and nb_goal == 1:
        return True
    else:
        return False
   ############################################################################################################################################

# Matrix conversion to have dimensions, start and goal positions
def conversion(matrix):

    # Put matrix into numpy array (in case not already)
    arr = np.array(matrix)
    
    # Find the indices of start and goal
    start_arr = np.argwhere(arr == 2)
    goal_arr = np.argwhere(arr == 3)
    
    # Get the height, width, and positions of start and goal
    max_val_x, max_val_y = arr.shape
    start = (start_arr[0][0], start_arr[0][1])
    goal = (goal_arr[0][0], goal_arr[0][1])

    # Replace the positions of 2 and 3 with 0
    arr[start] = 0
    arr[goal] = 0

    return max_val_x, max_val_y, start, goal, arr

############################################################################################################################################

# Creating the occupancy_grid 
def create_empty_plot(max_val_x, max_val_y):

    fig, ax = plt.subplots(figsize=(7,7)) 
    major_ticks_x = np.arange(0, max_val_x+1, 5)
    minor_ticks_x = np.arange(0, max_val_x+1, 1)
    major_ticks_y = np.arange(0, max_val_y+1, 5)
    minor_ticks_y = np.arange(0, max_val_y+1, 1)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val_y])
    ax.set_xlim([-1,max_val_x])
    ax.grid(True)
    
    return fig, ax

############################################################################################################################################

# Grow the obstacles to avoid collision
def grow_obstacles(matrix, growth_size):
    
    # Matrix to a numpy array (in case not already)
    arr = np.array(matrix)

    # New matrix with the same shape and filled with zeros
    expanded_matrix = np.zeros_like(arr)

    # Find the indices of obstacles (value = 1) in the original matrix
    obstacle_indices = np.where(arr == 1)

    # Grow obstacles in the expanded matrix
    for i, j in zip(obstacle_indices[0], obstacle_indices[1]):
        
        # Range for the expanded obstacles
        row_range = slice(max(0, i - growth_size), min(arr.shape[0], i + growth_size + 1))
        col_range = slice(max(0, j - growth_size), min(arr.shape[1], j + growth_size + 1))

        # Setting the corresponding elements to 1 in expanded matrix
        expanded_matrix[row_range, col_range] = 1

    return expanded_matrix, obstacle_indices

############################################################################################################################################

#Recurrently reconstructs the path from start node to the current node
def reconstruct_path(cameFrom, current):
 
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current]) 
        current=cameFrom[current]
        
    return total_path

############################################################################################################################################

# Algorithm to calculate the optimal path
def A_Star(start, goal, h, coords, occupancy_grid, max_val_x = 50, max_val_y = 50):

    
    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        assert point >= (0, 0) and point[0] < max_val_x and point[1] < max_val_y, "start or end goal not contained in the map"
    
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements
    s2 = math.sqrt(2)
    movements = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0), 
                (1, 1, s2), (-1, 1, s2), (-1, -1, s2), (1, -1, s2)]
    
    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]
    
    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            
            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            
            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue
                
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost
            
            if neighbor not in openSet:
                openSet.append(neighbor)
                
            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    
    return [], closedSet

############################################################################################################################################

def heuristics(max_val_x, max_val_y, goal):
    
    # List of all coordinates in the grid
    x,y = np.mgrid[0:max_val_x:1, 0:max_val_y:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles (Euclidean)
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))
    
    return h, coords

############################################################################################################################################

# Overall call with input: matrix and output: path, without displaying
def global_path(matrix):
    
    max_val_x, max_val_y, start, end, original_grid = conversion(matrix)

    # Grow the obstacles in the matrix
    growth_size = 9 # size of robot radius (in grid dimension) (would be 5.5cm)
    occupancy_grid, obstacle_indices = grow_obstacles(original_grid, growth_size)

    # Calling A*
    h, coords = heuristics(max_val_x, max_val_y, end)
    path, visitedNodes = A_Star(start, end, h, coords, occupancy_grid, max_val_x, max_val_y)
    path = np.array(path).reshape(-1, 2).transpose()
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
    
    return path, visitedNodes, occupancy_grid, obstacle_indices

############################################################################################################################################

# Display the matrix with path
def print_path(matrix, path, visitedNodes, occupancy_grid, obstacle_indices):
    
    max_val_x, max_val_y, start, end, arr = conversion(matrix)
    cmap = colors.ListedColormap(['white', 'orange'])
    
    # Displaying the map
    fig_astar, ax_astar = create_empty_plot(max_val_x, max_val_y)
    ax_astar.imshow(occupancy_grid.transpose(), cmap)

    # Plot the best path found and the list of visited nodes
    ax_astar.scatter(obstacle_indices[0], obstacle_indices[1], marker="o", color = 'red')
    ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'cyan')
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue')
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200)
    ax_astar.scatter(end[0], end[1], marker="o", color = 'purple', s=200)
    plt.title("best path in blue, visited nodes in orange")
    plt.title("Map : free cells in white, occupied cells in red")
    plt.show()
    
    return

############################################################################################################################################

#This function is to select only the keypoint of the path
def calculate_angles_along_path(path):
    key_points = [[], []]
    angle_TH = 45
    x_coords, y_coords = path[0], path[1]
    path_length = len(x_coords)

    if path_length < 3 or len(y_coords) != path_length:
        print("Invalid path format.")
        return angles

    for i in range(path_length - 2):
        if i % 10 == 0:
            kpoint = True
        point_A = [x_coords[i], y_coords[i]]
        point_B = [x_coords[i + 1], y_coords[i + 1]]
        point_C = [x_coords[i + 2], y_coords[i + 2]]

        angle_at_point = calculate_angle(point_A, point_B, point_C)
        if(angle_at_point > angle_TH) and kpoint == True:
            key_points[0].append(point_B[0])
            key_points[1].append(point_B[1])
            kpoint = False
    
    #We add the last point of the path as a Key point          
    key_points[0].append(x_coords[-1])
    key_points[1].append(y_coords[-1])
            
            
    return key_points

############################################################################################################################################

#Compute the angle between 3 points
def calculate_angle(A, B, C):
    
    vector_AB = np.array(B) - np.array(A)
    vector_BC = np.array(C) - np.array(B)

    dot_product = np.dot(vector_AB, vector_BC)
    magnitude_AB = np.linalg.norm(vector_AB)
    magnitude_BC = np.linalg.norm(vector_BC)

    cos_angle = dot_product / (magnitude_AB * magnitude_BC)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

############################################################################################################################################

def distance_between_points(point1, point2):
    
    return np.linalg.norm(np.array(point1) - np.array(point2))

############################################################################################################################################

def calculate_distances_to_keypoints(robot_position, key_points):
    distances = []

    for i in range(len(key_points[0])):
        distance = distance_between_points(robot_position, [key_points[0][i], key_points[1][i]])
        distances.append(distance)

    return distances

############################################################################################################################################

def find_closest_keypoint_index(robot_position, key_points):
    
    distances = calculate_distances_to_keypoints(robot_position, key_points)
    closest_index = np.argmin(distances)
    
    return closest_index      

############################################################################################################################################

def calculate_angle_between_robot_and_closest(robot_position, key_points, closest_index):
    
    closest_keypoint = [key_points[0][closest_index], key_points[1][closest_index]]
    angle = calculate_angle(closest_keypoint, robot_position, [0, 0])
    
    return angle 

############################################################################################################################################

def find_next_closest_keypoint(robot_position, key_points, distances, closest_index):
    angles = []
    
    for i in range(len(key_points[0])):
        if i != closest_index:
            angle = calculate_angle_between_robot_and_closest(robot_position, key_points, i)
            angles.append(angle)
        else:
            angles.append(360)  #Ensuring the closest key point isn't considered again
            
    next_closest_index = np.argmin(angles)
    
    return next_closest_index
    ###########################################################################################################################################
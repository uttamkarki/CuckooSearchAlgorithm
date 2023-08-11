import numpy as np
import matplotlib.pyplot as plt

#----------------- Hospital layout parameters ---------------- #
BedOrientation = 90
Obs = 20

BedOrientation_A = 1 if BedOrientation == 0 else 0
BedOrientation_B = 1 if BedOrientation == 90 else 0
BedOrientation_C = 1 if BedOrientation == 180 else 0
BedOrientation_D = 1 if BedOrientation == 270 else 0

ObstructionLevel_Low = 1 if Obs == 20 else 0
ObstructionLevel_Medium = 1 if Obs == 50 else 0
ObstructionLevel_High = 1 if Obs == 80 else 0

#

# -------------- initialize parameter for Cuckoo Search -------------#
n = 5  # number of nest
iteration = 100  # number of iteration
pa = 0.5  # egg detection probability

# ----------- defining function and bounds ------------#
x1_bound = [0, 500]
x2_bound = [0, 500]
n_decisionV = 2

g_best = np.zeros(n_decisionV)
g_best_obj = 0
g_best_pos = [0, 0]
history = []


# ------------- defining objective function calculation ------------------#

def fitness_function(solution):
    fitness = np.zeros(solution.shape[0])
    for i in range(solution.shape[0]):
        x1 = solution[i][0]
        x2 = solution[i][1]
        fx = 0.000494266380709747*x1*BedOrientation_A + 0.000494266380709747*x1*BedOrientation_C + 0.000494266380709747*x1*ObstructionLevel_Low + \
             0.000241788910901456*x2*ObstructionLevel_Low + np.arcsin(np.tan(0.0790296124852764*ObstructionLevel_High + 0.000494266380709747*x1 + 0.000279820914701597*x2)) \
             - 0.0790296124852764 - 0.000279820914701597*x2*ObstructionLevel_High - 0.000494266380709747*x1*ObstructionLevel_High


        if fx > 0.5:
            fx = fx- 0.25
        fitness[i] = fx
    return fitness


# ------------------------------------------------------------------------#

# -------------------- Get best solution between two solution ------------------------- #
def best_solution(solution1, solution2, fitness1, fitness2):
    for row in range(solution1.shape[0]):
        if fitness1[row] < fitness2[row]:
            nest_position[row][0] = solution1[row][0]
            nest_position[row][1] = solution1[row][1]
        else:
            nest_position[row][0] = solution2[row][0]
            nest_position[row][1] = solution2[row][1]
    return nest_position


# ---------------------------------------------------------------------#

# ------------------ get Global best value function -------------------- #
def get_global_best(solution, fitness, g_best):
    fitness_idx = np.where(fitness == np.max(fitness))[0][0]
    g_best[0] = solution[fitness_idx][0]
    g_best[1] = solution[fitness_idx][1]
    return g_best


# ----------------------------------------------------------------------#

# ---------------- Generate Cuckoo Solution ---------------------------#
def cuckoo_solution(host_position, s, g_best):
    for row in range(host_position.shape[0]):
        x1 = host_position[row][0]
        x2 = host_position[row][1]
        cuckoo_position[row][0] = max(0,x1 + 0.01 * np.random.uniform(0, 1) * s * (x1 - g_best[0]))
        cuckoo_position[row][1] = max(0,x2 + 0.01 * np.random.uniform(0, 1) * s * (x2 - g_best[1]))
    return cuckoo_position


# -------------------------------------------------------------------------#


pop_array = (n, n_decisionV)
host_position = np.random.uniform(low=0, high=1, size=pop_array)
cuckoo_position = np.random.uniform(low=0, high=1, size=pop_array)
nest_position = np.random.uniform(low=0, high=1, size=pop_array)
pos_after_detect = np.random.uniform(low=0, high=1, size=pop_array)
final_position = np.random.uniform(low=0, high=1, size=pop_array)

for solution in range(host_position.shape[0]):
    host_position[solution][0] = x1_bound[0] + np.random.uniform(low=0, high=1) * (x1_bound[1] - x1_bound[0])
    host_position[solution][1] = x2_bound[0] + np.random.uniform(low=0, high=1) * (x2_bound[1] - x2_bound[0])

for i in range(iteration):
    # ------- Calculating step size through levy flight distribution -------------- #
    beta = 1.5
    sigma_u = ((np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (
            np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.uniform(low=0, high=1) * sigma_u
    v = np.random.uniform(low=0, high=1)
    s = u / (v) ** (1 / beta)
    # -------------------------------------------------------------------------------#

    print("Iteration", i)
    host_obj = fitness_function(host_position)
    g_best = get_global_best(host_position, host_obj, g_best)

    # - getting cuckoo solution----- #

    cuckoo_position = cuckoo_solution(host_position, s, g_best)

    # --- getting cuckoo positon objective ---- #

    cuckoo_obj = fitness_function(cuckoo_position)
    nest_position = best_solution(host_position, cuckoo_position, host_obj, cuckoo_obj)
    nest_obj = fitness_function(nest_position)
    #print(nest_position)

    rand_pos_num = np.random.uniform(low=0, high=1, size=pop_array)
    rand_solution_num = np.random.randint(low=0, high=n, size=(n, 2))
    #print(rand_pos_num)
    #print(rand_solution_num)

    for row in range(rand_pos_num.shape[0]):
        if rand_pos_num[row][0] < pa:
            which_sol1 = rand_solution_num[row][0]
            which_sol2 = rand_solution_num[row][1]
            xd1 = nest_position[which_sol1][0]
            xd2 = nest_position[which_sol2][0]
            pos_after_detect[row][0] = max(0,nest_position[row][0] + np.random.uniform(0, 1) * (xd1 - xd2))
        else:
            pos_after_detect[row][0] = nest_position[row][0]
        if rand_pos_num[row][1] < pa:
            which_sol1 = rand_solution_num[row][0]
            which_sol2 = rand_solution_num[row][1]
            xd1 = nest_position[which_sol1][1]
            xd2 = nest_position[which_sol2][1]
            pos_after_detect[row][1] = max(0,nest_position[row][1] + np.random.uniform(0, 1) * (xd1 - xd2))
        else:
            pos_after_detect[row][1] = nest_position[row][1]

    obj_after_detect = fitness_function(pos_after_detect)
    #print(pos_after_detect)

    final_position = best_solution(nest_position, pos_after_detect, nest_obj, obj_after_detect)
    host_position = final_position
    #print(host_position)
    c_best_obj = np.max(fitness_function(host_position))
    #print("Iteration", i, "best Obj", c_best_obj)
    history.append(c_best_obj)
    if c_best_obj > g_best_obj:
        g_best_obj = c_best_obj
        index = (np.where(fitness_function(host_position) == np.max(fitness_function(host_position)))[0][0])
        g_best_pos[0] = host_position[index][0]
        g_best_pos[1] = host_position[index][1]
        #print(g_best_pos)
        #history.append(g_best_obj)

print("Best objective function: ", g_best_obj)
print("Best position: ", g_best_pos)
#print(history)
#plt.figure(figsize=(10, 8))
##plt.plot(history, linewidth=4)
#plt.show()

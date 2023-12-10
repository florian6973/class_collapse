import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform_direction
import numba
from numba import prange

np.random.seed(46) # 42 # 46

def compute(tau, N_per_class, classes, alpha, seed=46):
    def 
    np.random.seed(seed)
    u0 = uniform_direction(2).rvs(N_per_class*classes)
    result = minimize(objective_function, u0.reshape(-1), constraints={'type': 'eq', 'fun': constraint_unit_vector},
                    method='SLSQP', options={'disp': True})#, callback=update)
    # Extracting the optimized unit vector
    optimized_unit_vector = result.x.reshape(-1, 2)
    return spread_estimation(np.array(optimized_unit_vector))

tau = 0.1
N_per_class = 10
classes = 2

tau = 0.01 # more inf, very symmetric
N_per_class = 10
classes = 2

tau = 10 # not symmetric
N_per_class = 10
classes = 2

tau = 0.1 # bug?
N_per_class = 5
classes = 4

tau = 0.1 # bug?
N_per_class = 5
classes = 3

# tau = 1
# N_per_class = 10
# classes = 3
# alpha = 0.7#0.89187046#0.5#0.97

# Much more complex with three classes

tau = 0.1
N_per_class = 10
classes = 2
alpha = 0.85

# compute spread

u0 = uniform_direction(2).rvs(N_per_class*classes)
print(u0.shape)
# print(np.linalg.norm(u0, axis=1))
# exit()

np.set_printoptions(precision=3, suppress=True)

def spread_estimation(x):
    x = x.reshape(classes, N_per_class, 2)
    print(x)
    print(x.shape)
    avgs = np.mean(x, axis=1)
    print(avgs.shape)
    print(avgs)
    norms = np.linalg.norm(x - avgs.reshape(classes, -1, 2), axis=2)
    # print(x-avgs.reshape(3, -1, 2))
    print(norms.shape)
    spreads = np.mean(norms, axis=1)
    print(spreads.shape)
    return spreads

@numba.njit(parallel=True)
def supcon(x):    
    sum_i = 0
    for i in prange(len(x)):
        sum_loc = 0
        ui = x[i]
        class_i = i // N_per_class
        for p in prange(class_i*N_per_class, (class_i+1)*N_per_class):
            sum_sous = 0
            # sum_loc += 
            for a in prange(len(x)):
                # if a == i:
                #     continue
                if class_i*N_per_class <= a < (class_i+1)*N_per_class:
                    # a == i: see why important
                    sum_sous += 0
                else:
                    # sum_sous += np.exp(np.linalg.norm(ui - x[a])**2/tau)
                    sum_sous += np.exp(np.dot(ui, x[a])/tau)

            sum_sous = np.log(sum_sous)
            # sum_loc += np.linalg.norm(ui - x[p])**2/tau - sum_sous
            sum_loc += np.dot(ui, x[p])/tau - sum_sous
            

        # sum_loc += np.linalg.norm(x[i] - u0[i])**2
        sum_loc /= N_per_class        
        sum_i += sum_loc

    return -sum_i / len(x)

@numba.njit(parallel=True)
def nce(x):
    sum_i = 0
    for i in prange(len(x)):
        sum_loc = 0
        ui = x[i]
        class_i = i // N_per_class        
        for p in prange(class_i*N_per_class, (class_i+1)*N_per_class):
            sum_loc += np.exp(np.dot(ui, x[p])/tau)
        sum_loc = np.log(sum_loc)
        sum_i += np.dot(ui, ui)/tau - sum_loc

    return -sum_i / len(x)

# import numpy as np

def supcon2(x):
    sum_i = 0
    for i in range(len(x)):
        sum_loc = 0
        ui = x[i]
        class_i = i // N_per_class
        indices = np.arange(len(x))
        class_indices = indices[(class_i * N_per_class <= indices) & (indices < (class_i + 1) * N_per_class)]

        sum_sous = np.sum(np.exp(np.linalg.norm(ui - x[class_indices], axis=1)**2/tau))
        sum_sous = np.log(sum_sous)
        sum_loc += np.linalg.norm(ui - x[class_indices], axis=1)**2/tau - sum_sous

        sum_loc /= N_per_class
        sum_i += np.sum(sum_loc)

    return -sum_i / len(x)


print(supcon(u0))
# print(supcon2(u0))
# exit()
# print(supcon(u0))
# exit()

# plot u0
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
classes_labels = [i // N_per_class for i in range(len(u0))]
# plt.scatter(u0[:,0], u0[:,1], alpha=0.5, c=classes)
# plt.show()
# exit()

# Objective function to be minimized
def objective_function(x):
    x = x.reshape(-1, 2)
    return (1-alpha)*supcon(x)+alpha*nce(x)

# Constraint to ensure the vector is a unit vector

def constraint_rotation(x):
    # x = x.reshape(-1, 2)
    # return (x[0,0] - 1) ** 2 + (x[0,1] - 0) ** 2
    return 0
    
def constraint_unit_vector(x):
    x = x.reshape(-1, 2)
    # return np.concatenate((np.linalg.norm(x, axis=1) - 1.0, [constraint_rotation(x)]))
    return (np.linalg.norm(x, axis=1) - 1.0)

print(constraint_unit_vector(u0))
# exit()

# Initial guess
# initial_guess = np.random.rand(3)  # You can adjust the dimension based on your problem

def update(xk):
    print("Current Unit Vector:", xk.reshape(-1, 2))
    print("Current Value of the Objective Function:", objective_function(xk.reshape(-1, 2)))
    print()

# Optimization using scipy.optimize.minimize
result = minimize(objective_function, u0.reshape(-1), constraints={'type': 'eq', 'fun': constraint_unit_vector},
                    method='SLSQP', options={'disp': True}, callback=update)

# Extracting the optimized unit vector
optimized_unit_vector = result.x.reshape(-1, 2)
print(np.linalg.norm(optimized_unit_vector, axis=1))

print("Optimized Unit Vector:", optimized_unit_vector)
print("Optimized Value of the Objective Function:", result.fun)

# plt.rcParams["image.cmap"] = "Set1"
# # to change default color cycle
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

def add_circle():
    Drawing_uncolored_circle = plt.Circle( (0,0) ,
                                      1 ,
                                      fill = False )
    
    # axes.set_aspect( 1 )
    plt.gca().add_artist( Drawing_uncolored_circle )

def set_axis_lim():
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)

print(classes_labels)
print(spread_estimation(np.array(optimized_unit_vector)))
plt.subplot(1,2,1)
plt.scatter(u0[:,0], u0[:,1], alpha=0.9, c=classes_labels)
add_circle()
plt.axis('equal')
set_axis_lim()
plt.title("Before optimization")
plt.subplot(1,2,2)
plt.scatter(optimized_unit_vector[:,0], optimized_unit_vector[:,1], alpha=0.9, c=classes_labels)
add_circle()
plt.axis('equal')
set_axis_lim()
plt.title("After optimization")
plt.show()
exit()

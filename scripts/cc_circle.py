import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform_direction
import numba
from numba import prange
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

def compute(tau, N_per_class, classes, alpha, seed=46):        
    def objective_function(x):
        x = x.reshape(-1, 2)
        return (1-alpha)*supcon(x, N_per_class, tau)+alpha*nce(x, N_per_class, tau)
    
    def constraint_unit_vector(x):
        x = x.reshape(-1, 2)
        # return np.concatenate((np.linalg.norm(x, axis=1) - 1.0, [constraint_rotation(x)]))
        return (np.linalg.norm(x, axis=1) - 1.0)
        
    np.random.seed(seed)
    u0 = uniform_direction(2).rvs(N_per_class*classes)
    result = minimize(objective_function, u0.reshape(-1), constraints={'type': 'eq', 'fun': constraint_unit_vector},
                    method='SLSQP', options={'disp': True})#, callback=update)
    # Extracting the optimized unit vector
    optimized_unit_vector = result.x.reshape(-1, 2)
    return u0, optimized_unit_vector, spread_estimation(optimized_unit_vector, classes, N_per_class)

def plot_spread(tau, N_per_class, classes, alpha, seed=46):
    u0, optimized_unit_vector, spread = compute(tau, N_per_class, classes, alpha, seed)
    def add_circle():
        Drawing_uncolored_circle = plt.Circle( (0,0) ,
                                        1 ,
                                        fill = False )
        
        # axes.set_aspect( 1 )
        plt.gca().add_artist( Drawing_uncolored_circle )

    def set_axis_lim():
        plt.xlim(-1.3, 1.3)
        plt.ylim(-1.3, 1.3)

    plt.figure(figsize=(10, 10))
    classes_labels = [i // N_per_class for i in range(len(u0))]
    print(classes_labels)
    # print(spread_estimation(np.array(optimized_unit_vector)))
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


def spread_estimation(x, classes, N_per_class):
    x = x.reshape(classes, N_per_class, 2)
    avgs = np.mean(x, axis=1)
    norms = np.linalg.norm(x - avgs.reshape(classes, -1, 2), axis=2)
    spreads = np.mean(norms, axis=1)
    return spreads

@numba.njit(parallel=True)
def supcon(x, N_per_class, tau):    
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
def nce(x, N_per_class, tau):
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

plot_spread(0.1, 5, 2, 0.5)
for tau in [0.1, 0.5, 1]:
    alphas = np.linspace(0.5, 1, 10)
    sps = []
    for alpha in alphas:
        print(alpha)
        sps.append(compute(tau, 5, 2, alpha)[2])
    sps = np.array(sps)
    plt.plot(alphas, np.mean(sps, axis=1), label=f"tau={tau}")
plt.legend()
plt.show()

exit()

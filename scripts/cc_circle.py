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
        return (np.linalg.norm(x, axis=1) - 1.0)
        
    np.random.seed(seed)
    u0 = uniform_direction(2).rvs(N_per_class*classes)
    result = minimize(objective_function, u0.reshape(-1), constraints={'type': 'eq', 'fun': constraint_unit_vector},
                    method='SLSQP', options={'disp': True})
    
    optimized_unit_vector = result.x.reshape(-1, 2)
    return u0, optimized_unit_vector, spread_estimation(optimized_unit_vector, classes, N_per_class)


def plot_spread(tau, N_per_class, classes, alpha=None, alphas=None, seed=46):
    def add_circle():
        drawing_uncolored_circle = plt.Circle((0,0),
                                        1,
                                        fill = False)
        plt.gca().add_artist(drawing_uncolored_circle)

    def set_axis_lim():
        plt.xlim(-1.3, 1.3)
        plt.ylim(-1.3, 1.3)

    if alpha is not None:
        u0, optimized_unit_vector, spread = compute(tau, N_per_class, classes, alpha, seed)
        plt.figure(figsize=(10, 10))
        classes_labels = [i // N_per_class for i in range(len(u0))]
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
    elif alphas is not None:
        n = len(alphas) + 1
        rows = n // 2 + n % 2
        fig, axs = plt.subplots(rows, 2, figsize=(10, 5*rows))
        axs = axs.ravel()  # Flatten the array of axes
        for i, alpha in enumerate(alphas, start=1):  # Start from 1 to leave the first subplot for "Before optimization"
            u0, optimized_unit_vector, spread = compute(tau, N_per_class, classes, alpha, seed)
            if i == 1:
                classes_labels = [i // N_per_class for i in range(len(u0))]
                axs[0].scatter(u0[:,0], u0[:,1], alpha=0.9, c=classes_labels, cmap='flag', s=200)
                Drawing_uncolored_circle = plt.Circle((0,0), 1, fill=False)
                axs[0].add_artist(Drawing_uncolored_circle)
                axs[0].set_title("Before optimization")
                axs[0].set_xlim(-1.3, 1.3)
                axs[0].set_ylim(-1.3, 1.3)
            axs[i].scatter(optimized_unit_vector[:,0], optimized_unit_vector[:,1], alpha=0.9, 
                        c=classes_labels, cmap='flag', s=200)
            Drawing_uncolored_circle = plt.Circle((0,0), 1, fill=False)
            axs[i].add_artist(Drawing_uncolored_circle)
            axs[i].set_title(f"alpha={alpha}")
            axs[i].set_xlim(-1.3, 1.3)
            axs[i].set_ylim(-1.3, 1.3)
            axs[i].legend()
        plt.tight_layout()
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
            for a in prange(len(x)):
                if class_i*N_per_class <= a < (class_i+1)*N_per_class:
                    # a == i: see why important
                    sum_sous += 0
                else:
                    sum_sous += np.exp(np.dot(ui, x[a])/tau)

            sum_sous = np.log(sum_sous)
            sum_loc += np.dot(ui, x[p])/tau - sum_sous
            
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

# Plot vectors before and after optimization
plot_spread(0.1, 10, 2, alphas=[0, 0.9, 1])

# Plot spread for different tau
for tau in [0.1, 0.25, 0.5, 1, 2]:
    alphas = np.linspace(0.5, 1, 10)
    sps = []
    for alpha in alphas:
        print(alpha)
        sps.append(compute(tau, 10, 2, alpha)[2])
    sps = np.array(sps)
    plt.plot(alphas, np.mean(sps, axis=1), label=f"tau={tau}")

plt.xlabel("alpha")
plt.ylabel("spread")
plt.legend()
plt.show()
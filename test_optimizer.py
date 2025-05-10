import cProfile
import pstats
from network.optimizers import AdamOptimizer
from network.gradient_tape import GradientTape
from network.types import Variable
import numpy as np

# Testing on f(w) = (w - 3)^2
target = np.array([1 + 2j, 3 + 4j])
w = Variable(value=[0.0+0.0j, 0.0+0.0j], shape=(2,), dtype=np.complex64, trainable=True, name='w', initializer='zeros')
opt = AdamOptimizer(learning_rate=1e-1)
steps = int(1e4)

def func(x):
    error = (x - target) * np.conj(x - target)
    error *= np.cos(x) / np.sqrt(np.pi)
    return error

def mse(x):
    return np.mean((x - target) * np.conj(x - target)).real

def run_optimization():
    print("Step\tw value")
    for i in range(steps):
        with GradientTape() as tape:
            tape.watch(w)
            out = func(w)
            
        loss = mse(out)
            
        grad = tape.gradient(loss, w)
        # print(f"Grad: {grad}")
        opt.apply_gradients([(grad, w)])
        if i % (steps // 100) == 0:
            print(f"{i+1}\t{w}\t{loss}")
            
        if not loss:
            print(f"{i+1}\t{w}\t{loss}")
            print(f"Converged at step {i+1}")
            break

    print(f"\nFinal parameter value after {steps} steps: {w}")

# Profile the function using cProfile
pr = cProfile.Profile()
pr.enable()

run_optimization()

pr.disable()

# Create a stats object
stats = pstats.Stats(pr)
stats.sort_stats('time')  # Sorting by time spent
stats.print_stats(3)  # Display the top 3 most time-consuming functions

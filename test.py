from network.types.variable import Variable
import numpy as np

# Correct way to create an instance: Pass arguments matching __new__
# Notice the order and meaning of arguments
a = Variable(shape=(1, 1), dtype=np.float16, trainable=True, name="my_variable_a", initializer="random_normal")
b = Variable(shape=(2, 3), dtype=np.float32, trainable=False, name="another_var")

a.initialize()
b.initialize()

print(f"Variable '{a.name}': shape={a.shape}, dtype={a.dtype}, trainable={a.trainable}, value={a}")
print(f"Variable '{b.name}': shape={b.shape}, dtype={b.dtype}, trainable={b.trainable}, value={b}")

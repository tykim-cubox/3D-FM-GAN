import ray
import time

# Start Ray.
ray.init()

gpus=8
@ray.remote(num_gpus=gpus, max_calls=gpus)
def f(x):
    time.sleep(0.0001)
    return x

# Start 4 tasks in parallel.
result_ids = []
for i in range(4):
    result_ids.append(f.remote(i))
    
# Wait for the tasks to complete and retrieve the results.
# With at least 4 cores, this will take 1 second.
results = ray.get(result_ids)  # [0, 1, 2, 3]
print(results)
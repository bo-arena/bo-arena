# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType

from bo_arena.benchmarks import BencherBenchmark

# %%
# Example taken from bencher README: https://github.com/lpapenme/bencher

# Create a client to communicate with the Bencher server
# By default, it connects to 127.0.0.1:50051
client = BencherClient()

# Create a list of values to evaluate
point_one = [Value(type=ValueType.CONTINUOUS, value=0.5) for _ in range(180)]
point_two = [Value(type=ValueType.CONTINUOUS, value=0.1) for _ in range(180)]
# The benchmark name is the name of the benchmark you want to evaluate
benchmark_name = "lasso-dna"

# Evaluate the benchmark with the given values
# This will send the values to the server and return the result
# If the server is not running, it will raise an error
result_one = client.evaluate_point(
    benchmark_name=benchmark_name,
    point=point_one,
)
result_two = client.evaluate_point(
    benchmark_name=benchmark_name,
    point=point_two,
)
print(f"Result one: {result_one}")
print(f"Result two: {result_two}")

# %%
# Now using the bo-arena BencherBenchmark wrapper

lasso_benchmark = BencherBenchmark(name="lasso-dna")
point_one = np.array([0.5] * 180).reshape(1, -1)
point_two = np.array([0.1] * 180).reshape(1, -1)
points = np.vstack([point_one, point_two])
res = lasso_benchmark.evaluate_points(points)
print(f"Result one: {res[0]}")
print(f"Result two: {res[1]}")

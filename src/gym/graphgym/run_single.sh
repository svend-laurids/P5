#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
uv run main.py --cfg configs/pyg/example_node.yaml --repeat 3 # node classification
uv run main.py --cfg configs/pyg/example_link.yaml --repeat 3 # link prediction
uv run main.py --cfg configs/pyg/example_graph.yaml --repeat 3 # graph classification

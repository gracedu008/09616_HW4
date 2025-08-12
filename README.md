# 09616 - Neural Networks & Deep Learning in Science: HW4 - Graph Neural Networks

## Description
A graph neural network (GNN) is a type of neural network that is specifically designed to work with graph data structures. Graphs are mathematical structures that consist of nodes (also known as vertices) and edges (also known as links) that connect them. GNNs can learn to extract features from graphs and make predictions based on them.

In molecular science, GNNs have become increasingly popular for tasks such as predicting molecular properties, discovering new molecules, and designing new drugs. Molecules can be represented as graphs, where atoms are represented as nodes and bonds between atoms are represented as edges. GNNs can be trained on large datasets of molecular graphs and their associated properties, allowing them to learn to predict properties of new molecules that they have not seen before.

In this Homework, you are required to implement a graph-based neural network model to predict a target property (Regression). The train and test datasets have been provided to you. You will train a GNN on the train dataset, then use it to make predictions on the test dataset. This Homework specifically requires implementation using PyTorch Geometric (PyG): https://pytorch-geometric.readthedocs.io/en/latest/

## Evaluation
The evaluation metric for this competition is the mean absolute error regression loss (sklearn.metrics.mean_absolute_error).

### My submission achieved 0.53476 in mean absolute error regression loss.

# Geometric learning and phylogenetics
Will the Geometric DL approach will work with phylogenetics?

## Background

*Phylogeny reconstruction* is a very useful tool in many fields of biology. It can be used for representing the evolutionary history of a taxon, sequence function inference and even finding a source of infection.

There are many different methods of phylogeny reconstruction and it is difficult to choose appropriate for your sequences.

Existing quality measures are not ideal. *Bootstrap* is slow, *likelihood* of a branch can be used only with *maximum-likelihood* method, etc.

Recently a few deep learning models have been introduced in the field. They are fast and powerful but works with sequences only.

Geometric deep learning techniques allow us to learn directly on graph data, e.g. a tree. The idea of spatial convolution on a graph is to update features of a given node using features of adjacent nodes. It may be considered as a generalisation of CNN, which is widely used in computer vision.

## Methods

#### What are we going to predict

Every node, unless it's a leaf, splits a tree into three parts. So, to compare two nodes in different trees we use standard metric for k-partitions from a [paper](https://doi.org/10.1016/j.dam.2010.09.002).
For every node in the given tree, we find the closest node in the target tree and save the distance to use it as a label of the node.

#### What are the features

Input trees of the model should contain features in the nodes. We have used max difference of the residue frequencies in groups of the 3-partition. It may be very hard to understand, so the formula should help. For i-th node and j-th pair of residuals feature value is:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\fn_cm&space;\tiny&space;x_{i,j}&space;=&space;max_{k\in&space;Adjacent-to(i)}|freq(a_1,leafs-from-parent(k))-freq(a_2,leafs-from-parent(i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;\fn_cm&space;\tiny&space;x_{i,j}&space;=&space;max_{k\in&space;Adjacent-to(i)}|freq(a_1,leafs-from-parent(k))-freq(a_2,leafs-from-parent(i))" title="\tiny x_{i,j} = max_{k\in Adjacent-to(i)}|freq(a_1,leafs-from-parent(k))-freq(a_2,leafs-from-parent(i))" /></a>

#### Model and data

Model is simple: 2 GIN layers, which can perform some feature transformation with their incorporated multilayered perceptron, then 1 SAGE layer, which is averaging adjacent features.

Our training dataset contained 3000 trees of fungi and 2800 trees of archaea. Surprisingly, the model was small enough to train it on a single laptop GPU.

The training was performed with "reduce-on-plateau" learning rate scheduler.

## Reproduce me!
First, you need some data. Our model using [phylobench](http://mouse.belozersky.msu.ru/phylobench/) If you want to make your own dataset from alignments see graph_data comments.
The main file to run is notebooks/support_values.ipynb. 
More sources, data, and running guide are coming soon!

## Assessing whole trees

A notebook `trees` contain a new model for tree assessment. It requires different data format, class `MAGraph` can construct the data from alignments, nwk trees and an etalon tree. 

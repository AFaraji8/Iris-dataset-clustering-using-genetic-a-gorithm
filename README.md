This project implements a Genetic Algorithm (GA) from scratch to perform clustering on the classic UCI Iris Dataset. 

Unlike traditional clustering methods like K-Means, this approach uses evolutionary strategies—such as mutation and crossover—to minimize the intra-cluster distance and find the optimal labels for each data point.

📌 Project Overview
        The goal is to partition 150 Iris flower samples into $K$ clusters (specifically $K=3$ for the three species) by evolving a population of potential solutions (chromosomes).
Key Components:

        Chromosome Representation:
                Each chromosome is an array where each index represents a data point and its value represents the assigned cluster.
                
        Fitness Function: 
                Calculated based on the Euclidean distance between data points and their respective cluster centroids (Objective Function).
                
        Evolutionary Operators:- Crossover: 
                Swapping cluster assignments between two parent chromosomes.
                
        Mutation: 
                Re-assigning random data points to the nearest centroid to speed up convergence.
                
        Accuracy Mapping: 
                Since clustering is unsupervised, the code includes a mapping function to align discovered clusters with the actual species labels for accuracy evaluation.

from ucimlrepo import fetch_ucirepo 
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#___________________________________________preparing data_____________________________________________
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features              #[150 rows x 1 columns]
y = iris.data.targets 
  
# metadata 
#print(iris.metadata) 
  
# variable information 
print(iris.variables) 
print("\n\n\n X:")
print(X)
print("\n Y:")
print(y)

"""
numeric_features = X.select_dtypes(include=['float64', 'int64'])
print(numeric_features)
"""



num_models = len(y['class'].value_counts())         #num_models=3

print(" number of models of flowers:", num_models)


#Initialize frist population (random) for start algorithm

mychromosome=[]
for i in range(4):
    chromosomes = list(np.random.choice([0, 1, 2], size=151))
    chromosomes[150]=3
    mychromosome.append(chromosomes)

#print(mychromosome)



label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y['class'])
#print("y_numeric:")
#print(y_numeric)




#___________________________________________functions_____________________________________________


def calculate_centroid(data, labels, cluster_index):
    cluster_data = data[np.array(labels) == cluster_index]
    centroid = cluster_data.mean(axis=0)
    return centroid





def objective_function(data, labels, num_clusters):
    total_distance = 0
    centercluster = []
    for x in range(num_clusters):
        centercluster.append(calculate_centroid(data, labels[:-1], x))

    for i in range(len(data)):
        total_distance += np.linalg.norm(data.iloc[i].to_numpy() - centercluster[labels[i]], 2)
    return total_distance





def mutation(chromosome, num_clusters, data):
    mutated_chromosome = chromosome.copy()
    indices_to_mutate = np.random.choice(len(chromosome)-1, size=20, replace=False)

    for index in indices_to_mutate:
        #print(index)
        data_point = data.iloc[index]
        distances = [np.linalg.norm(data_point - calculate_centroid(data, chromosome[:-1], cluster)) for cluster in range(num_clusters)]
        new_cluster = np.argmin(distances)
        mutated_chromosome[index] = new_cluster

    return mutated_chromosome



def crossover(parent1, parent2):
    crossover_points = np.random.choice(len(parent1)-1, size=50, replace=False)
    child1 = parent1.copy()
    child2 = parent2.copy()
    for point in crossover_points:
        #print(point)
        child1[point], child2[point] = child2[point], child1[point]

    return child1, child2



def genetic_algorithm(data, num_clusters, num_generations, population_flist):
    population = population_flist

    for generation in range(num_generations):
        new_population = population.copy()
        for chromosome in population:
            mutated_chromosome = mutation(chromosome, num_clusters, data)
            new_population.append(mutated_chromosome)
            p=random.choice(new_population)
            crossover_chromosome = crossover(chromosome,p)
            new_population.extend(crossover_chromosome)

        best_chromosomes = select_best_individuals(data, new_population)
        population= best_chromosomes

    return get_best_clusters(data, population)



def select_best_individuals(data, population):
    best_chromosomes = []
    for chromosome in population:
        labels = np.array(chromosome)
        fitness = objective_function(data, labels,chromosome[-1])
        best_chromosomes.append((chromosome, fitness))

    best_chromosomes.sort(key=lambda x: x[1])
    best_chromosomes = [chromosome for chromosome, _ in best_chromosomes]
    return best_chromosomes[0:4]



def get_best_clusters(data, chromosomess):
    best_clusters = None
    best_fitness = float('inf')

    for chromosome in chromosomess:
        labels =chromosome
        fitness = objective_function(data, labels,chromosome[-1])

        if fitness < best_fitness:
            best_fitness = fitness
            best_clusters = labels

    return best_clusters





def find_best_mapping(g, y_numeric):
    unique_clusters = np.unique(g)
    unique_classes = np.unique(y_numeric)
    
    best_mapping = {}
    best_correct_count = 0

    for cluster in unique_clusters:
        cluster_indices = np.where(g == cluster)[0]
        cluster_classes = y_numeric[cluster_indices]

        correct_mapping = None
        max_correct_count = 0

        for c in unique_classes:
            correct_count = np.sum(cluster_classes == c)
            if correct_count > max_correct_count:
                max_correct_count = correct_count
                correct_mapping = c

        best_mapping[cluster] = correct_mapping
        best_correct_count += max_correct_count

    accuracy = best_correct_count / len(g)
    return best_mapping, accuracy







#___________________________________________main_____________________________________________


"""

print(" \n for different k :")

for i in range(2,11):
    xchromosomes = list(np.random.choice(list(range(i)), size=151))
    xchromosomes[150]=i
    x1chromosomes = list(np.random.choice(list(range(i)), size=151))
    x1chromosomes[150]=i
    gx=genetic_algorithm(X,i,25,[xchromosomes,x1chromosomes])
    accuracyxx= accuracy_score(y_numeric, gx[:-1])
    print("k=",i,"    Accuracy:", accuracyxx)
    print(" ")

print(" \n\n\n\n")

"""

print(" \n\n\n")

print("for the best number of clusters (k=3):\n")



#print("f:",mychromosome)

g=genetic_algorithm(X,3,50,[mychromosome[0],mychromosome[1]])
print("g:",g)



best_mapping, accuracy = find_best_mapping(g[:-1], y_numeric)
#print("Best Mapping:", best_mapping)

y_strings =np.unique(label_encoder.inverse_transform(y_numeric))
     
#print("y_strings:", y_strings)

print("Best Mapping:")
for cluster, correct_class in best_mapping.items():
    print(f"Cluster {cluster} is mapped to class {y_strings[correct_class]}")

print("Accuracy:", accuracy)

print(" ")




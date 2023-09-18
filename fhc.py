from asyncio import ALL_COMPLETED
import random as random
import math as math
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
import time as time
import numpy as np
#import matplotlib

#####
#code for the edge-weighted graph, I used resources online to create the edgeweighted graph, and then modified it for my use, since it isn't really the focus of the project/algorithm, just the dataset itself
#this creates the toy problem
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent to: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_edges(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()
    
    def get_dict(self):
        return self.vert_dict

    #def create_data(self, size): make it a real program later

class FoolishHillClimbing:
    def __init__(self):
        self.pop = None
        self.graph = None
        
    def initialSolution(self, probSize):
        chrom = []
        while len(chrom) < probSize:
            attempt = random.randint(0, probSize-1)
            if attempt not in chrom: chrom.append(attempt)
        return chrom
  
    def fitness(self, k, g, prSize, chromosome):
        chrom = chromosome
        numK = k
        probSize = prSize
        graph = g
        fitnessScore = 0
        v = Vertex
        #w = Vertex
        
        vertices = graph.get_dict()
        #print(vertices)
        
        for i in range(numK, probSize):
            v = vertices[str(chrom[i])]
            w = vertices[str(chrom[int(i/numK)-1])]
            fitnessScore += v.get_weight(w)
        return fitnessScore

    #mutation methods
    def pairwise_exchange(self, chrom, prSize):
        probSize = prSize
        mutatedChromosome = chrom
        index1, index2 = random.randint(0, probSize-1), random.randint(0, probSize-1)
        mutatedChromosome[index1], mutatedChromosome[index2], = mutatedChromosome[index2], mutatedChromosome[index1]
        return mutatedChromosome
         
    def nn_mutate(self, chrom, prSize, graph, kNum):
        probSize = prSize
        candidateChromosome = chrom
        
        originalChromosome = candidateChromosome.copy()
        
        g = graph
        k = kNum
        
        vertices = g.get_dict()
        
        for allele in candidateChromosome:  
            v = vertices[str(allele)]
            nearest = 200
            nearestNode = g.get_vertex(str(random.randint(0,probSize-1)))
            for w in v.get_edges():
                if v.get_weight(w) < nearest:
                    nearestNode = w
                    nearest = v.get_weight(w)
            candidateChromosome[allele], candidateChromosome[int(nearestNode.id)] = candidateChromosome[int(nearestNode.id)], candidateChromosome[allele]
            if FoolishHillClimbing.fitness(self, k, g, probSize, candidateChromosome) < FoolishHillClimbing.fitness(self, k, g, probSize, originalChromosome):
                candidateChromosome = originalChromosome
            #originalChromosome = candidateChromosome
        return candidateChromosome

if __name__ == '__main__':
    #constants/variables
    solution = []
    newSolution = []
    initialTemperature = 20.0
    initialP = 1000
    alpha = .98
    beta = 1.02
    
    #toy data
    #problemSize = 20
    #kNum = 4
    
    #small data
    #problemSize = 50
    #kNum = 5
    
    #medium data
    problemSize = 100
    kNum = 10
    
    fitnesses = []
    sa = FoolishHillClimbing()
    #toy data
    #x=[10, 0, 0, 20, 20, 10, 0, 0, 20, 20, 90, 80, 80, 100, 100, 90, 80, 80, 100, 100]
    #y=[10, 0, 20, 0, 20, 90, 80, 100, 80, 100, 10, 0, 20, 0, 20, 90, 80, 100, 80, 100]
    #[0,5,10,15,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
    
    #small data
    #x= [78, 80, 1, 4, 0, 1, 49, 1, 37, 4, 20, 86, 77, 40, 24, 1, 45, 87, 22, 0, 57, 97, 40, 57, 61, 18, 35, 43, 93, 72, 54, 38, 13, 90, 80, 75, 79, 79, 81, 72, 8, 65, 48, 26, 15, 48, 88, 20, 88, 90]
    #y= [87, 49, 30, 1, 75, 92, 45, 2, 34, 53, 94, 47, 90, 37, 88, 84, 31, 25, 80, 89, 46, 29, 59, 20, 79, 14, 56, 31, 48, 45, 34, 59, 49, 58, 10, 83, 70, 93, 35, 61, 91, 43, 77, 37, 88, 5, 42, 10, 59, 0]

    #medium data
    x = [17, 15, 6, 66, 76, 20, 76, 52, 73, 65, 27, 15, 22, 42, 17, 53, 20, 28, 32, 76, 18, 26, 3, 76, 81, 24, 1, 91, 67, 24, 47, 4, 92, 43, 88, 51, 44, 26, 89, 15, 88, 23, 38, 40, 53, 11, 4, 38, 89, 22, 22, 93, 57, 57, 78, 84, 15, 71, 59, 16, 39, 2, 14, 49, 65, 100, 0, 16, 88, 13, 83, 93, 64, 0, 20, 59, 69, 33, 69, 96, 73, 35, 40, 49, 50, 29, 46, 22, 0, 91, 86, 8, 95, 83, 95, 4, 89, 16, 81, 2]
    y = [58, 46, 34, 7, 48, 25, 85, 78, 57, 86, 2, 81, 7, 62, 90, 3, 24, 61, 1, 27, 32, 56, 55, 46, 13, 89, 91, 42, 34, 21, 25, 12, 54, 26, 98, 11, 43, 99, 10, 53, 92, 78, 89, 97, 16, 74, 1, 13, 78, 63, 34, 69, 29, 15, 87, 61, 99, 27, 66, 98, 49, 3, 20, 46, 50, 18, 94, 2, 41, 80, 50, 33, 75, 27, 50, 46, 16, 65, 19, 83, 79, 92, 52, 19, 49, 9, 71, 2, 19, 83, 34, 12, 57, 49, 38, 37, 62, 86, 8, 33]

    #create graph for data/coordinates
    g = Graph()
    n=0
    for n in range(0, problemSize):
        g.add_vertex(str(n))
        for existing in range(0, g.num_vertices):
            if g.num_vertices <= 1: 
                ""
            else:
                g.add_edge(str(existing), str(n), math.sqrt(float((math.pow(abs(x[n]-x[existing]), 2) + math.pow(abs(y[n]-y[existing]), 2)))))
    ##########sa
    
    #initiate
    solution = sa.initialSolution(problemSize)
    #print(iSolution)
    temperature = initialTemperature
    iterations = initialP
    startTime = time.time()
    
    #loop the SA
    while temperature > 1:
        for i in range(0, iterations):
            fitness = sa.fitness(kNum, g, problemSize, solution)
            #print(fitness)
            
            #PERTURBATION
            newSolution = sa.pairwise_exchange(solution, problemSize)
            #newSolution = sa.nn_mutate(solution, problemSize, g, kNum)
            newFitness = sa.fitness(kNum, g, problemSize, newSolution)
            if (newFitness < fitness): #or (random.random() < math.pow(math.e,(fitness-newFitness)/temperature)):
                solution = newSolution
        temperature *= alpha
        iterations =int(iterations*beta)
        fitness = sa.fitness(kNum, g, problemSize, solution)     
    fitness = sa.fitness(kNum, g, problemSize, solution)     
    print(solution)   
    print(fitness)
    print("--- %s seconds ---" % (time.time() - startTime))
from asyncio import ALL_COMPLETED
import random as random
import math as math
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

class GeneticAlgorithm:
    def __init__(self):
        self.pop = None
        self.graph = None
        
    def initialPopulationRandom(self, probSize, popSize):
        pop = []
        chrom = []
        for i in range(popSize):
            while len(chrom) < probSize:
                attempt = random.randint(0, probSize-1)
                if attempt not in chrom: chrom.append(attempt)
            pop.append(chrom)
            chrom = []
        return pop
  
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
        #punish infeasibles
        if len(chrom) != len(set(chrom)):
            fitnessScore = fitnessScore * 10
        return fitnessScore
    
    def bestFitness(self, fits):
        fitness = fits
        bestFit = 10000
        bestFitIndex = -1
        for i in range(0, len(fitness)):
            if fitness[i] < bestFit:
                bestFit = fitness[i]
                bestFitIndex = i
        return bestFitIndex
    
    def elites(self, fits, popSize):
        fitness = fits
        first = 0
        second = 1
        for c in range(popSize):
            if fitness[c] < fitness[first]:
                first = c
            elif fitness[c] < fitness[second]:
                second = c
        return first, second
    
    #selection methods 
    def tournament(self, fits, pSize):
        fitness = fits
        popSize = pSize
        winner = 0
        cand1, cand2 = random.randint(0, popSize - 1), random.randint(0, popSize - 1)
        c1Fit = fitness[cand1]
        c2Fit = fitness[cand2]
        winner = cand1 if (c1Fit < c2Fit) else cand2 if (c2Fit < c1Fit) else cand1
        #print(cand1, c1Fit)
        #print(cand2, c2Fit)
        #print(winner)
        return winner
    
    def roulette(self, fits, pSize, cPop):
        fitness = fits
        popSize = pSize
        currPop = cPop
        newPop = []
        probs = []
        max = sum(fitness)
        for fit in fitness:
            probs.append(fit/max)
        probs = 1 - np.array(probs)
        while len(newPop) < popSize:
            pick = random.uniform(0,max)
            current = 0
            for i in range(0, len(currPop)-1):
                current += probs[i]
                if current > pick:
                    newPop.append(currPop[i])
        return newPop
        
    #crossover methods
    def singlepoint_crossover(self, parPop, poSize, prSize, rate):
        currPop = parPop
        popSize = poSize
        probSize = prSize
        crossRate = rate
        split = 0
        
        child1 = []
        child2 = []
        p1 = random.randint(0, len(currPop)-1)
        p2 = random.randint(0, len(currPop)-1)
        newChildren = [currPop[p1], currPop[p2]]
        split = random.randint(0, probSize)
        
        if (p1 != p2) & (random.randint(0, 100) <= crossRate):
            parent1, parent2 = currPop[p1], currPop[p2]
            for i in range(0, split):
                child1.append(parent1[i])
                child2.append(parent2[i])
            for i in range(split, probSize):
                if parent1[i] not in child2:
                    child2.append(parent1[i])
                if parent2[i] not in child1:
                    child1.append(parent2[i])
            #fix children at end
            while len(child1) < probSize:
                attempt = random.randint(0, probSize-1)
                if attempt not in child1: child1.append(attempt)    
            while len(child2) < probSize:
                attempt = random.randint(0, probSize-1)
                if attempt not in child2: child2.append(attempt)
            newChildren[0] = (child1)
            newChildren[1] = (child2)
        parents = [currPop[p1], currPop[p2]]
        
        return newChildren
    
       
    def vector_crossover(self, parPop, poSize, prSize, rate):
        currPop = parPop
        popSize = poSize
        probSize = prSize
        crossRate = rate
        
        child1 = []
        child2 = []
        p1 = random.randint(0, len(currPop)-1)
        p2 = random.randint(0, len(currPop)-1)
        newChildren = [currPop[p1], currPop[p2]]
        crossVector = []
        
        for i in range(1, random.randint(0, probSize-1)):
            crossVector.append(random.randint(0,probSize))
        
        if (p1 != p2) & (random.randint(0, 100) <= crossRate):
            parent1, parent2 = currPop[p1], currPop[p2]
            for i in parent1:
                if i in crossVector:
                    if parent1[i] not in child2:
                        child2.append(parent1[i])
                    if parent2[i] not in child1:
                        child1.append(parent2[i])
                else:
                    if parent1[i] not in child1:
                        child1.append(parent1[i])
                    if parent2[i] not in child2:
                        child2.append(parent2[i])
            #fix children at end
            while len(child1) < probSize:
                attempt = random.randint(0, probSize-1)
                if attempt not in child1: child1.append(attempt)    
            while len(child2) < probSize:
                attempt = random.randint(0, probSize-1)
                if attempt not in child2: child2.append(attempt)
            newChildren[0] = (child1)
            newChildren[1] = (child2)
        return newChildren
    
    #mutation methods
    def pairwise_exchange(self, chrom, prSize, mutRate):
        probSize = prSize
        mutatedChromosome = chrom
        if(random.randint(0,100) < mutRate):
            index1, index2 = random.randint(0, probSize-1), random.randint(0, probSize-1)
            mutatedChromosome[index1], mutatedChromosome[index2], = mutatedChromosome[index2], mutatedChromosome[index1]
        return mutatedChromosome
         
    def nn_mutate(self, chrom, prSize, mutRate, graph, kNum):
        probSize = prSize
        candidateChromosome = chrom
        
        originalChromosome = candidateChromosome.copy()
        
        g = graph
        k = kNum
        
        vertices = g.get_dict()
        
        if(random.randint(0,100) < mutRate):
            for allele in candidateChromosome:  
                v = vertices[str(allele)]
                nearest = 200
                nearestNode = g.get_vertex(str(random.randint(0,probSize-1)))
                for w in v.get_edges():
                    if v.get_weight(w) < nearest:
                        nearestNode = w
                        nearest = v.get_weight(w)
                candidateChromosome[allele], candidateChromosome[int(nearestNode.id)] = candidateChromosome[int(nearestNode.id)], candidateChromosome[allele]
                if GeneticAlgorithm.fitness(self, k, g, probSize, candidateChromosome) > GeneticAlgorithm.fitness(self, k, g, probSize, originalChromosome):
                    candidateChromosome = originalChromosome
            #originalChromosome = candidateChromosome
        return candidateChromosome
                        
                        

"""
    #extra
    def simulated_annealing(self):
        None
        
    def foolish_hill_climbing(self):
        None
"""
    

if __name__ == '__main__':
    #constants/variables
    population = []
    childPopulation = []
    parentPopulation = []
    mutatedChildPopulation = []
    chromosome = []
    generation = 0
    crossoverRate = 90
    mutationRate = 10
    populationSize = 100
    #toy data
    problemSize = 20
    kNum = 4
    
    #small data
    #problemSize = 50
    #kNum = 5
    
    #medium data
    #problemSize = 100
    #kNum = 10
    
    fitnesses = []
    generations = 0
    maxGen = 10000
    eIndex1 = 0
    eIndex2 = 0
    ga = GeneticAlgorithm()
    bestFitness = 100000
    #toy data
    x=[10, 0, 0, 20, 20, 10, 0, 0, 20, 20, 90, 80, 80, 100, 100, 90, 80, 80, 100, 100]
    y=[10, 0, 20, 0, 20, 90, 80, 100, 80, 100, 10, 0, 20, 0, 20, 90, 80, 100, 80, 100]
    #[0,5,10,15,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
    
    #small data
    #x= [78, 80, 1, 4, 0, 1, 49, 1, 37, 4, 20, 86, 77, 40, 24, 1, 45, 87, 22, 0, 57, 97, 40, 57, 61, 18, 35, 43, 93, 72, 54, 38, 13, 90, 80, 75, 79, 79, 81, 72, 8, 65, 48, 26, 15, 48, 88, 20, 88, 90]
    #y= [87, 49, 30, 1, 75, 92, 45, 2, 34, 53, 94, 47, 90, 37, 88, 84, 31, 25, 80, 89, 46, 29, 59, 20, 79, 14, 56, 31, 48, 45, 34, 59, 49, 58, 10, 83, 70, 93, 35, 61, 91, 43, 77, 37, 88, 5, 42, 10, 59, 0]

    #medium data
    #x = [17, 15, 6, 66, 76, 20, 76, 52, 73, 65, 27, 15, 22, 42, 17, 53, 20, 28, 32, 76, 18, 26, 3, 76, 81, 24, 1, 91, 67, 24, 47, 4, 92, 43, 88, 51, 44, 26, 89, 15, 88, 23, 38, 40, 53, 11, 4, 38, 89, 22, 22, 93, 57, 57, 78, 84, 15, 71, 59, 16, 39, 2, 14, 49, 65, 100, 0, 16, 88, 13, 83, 93, 64, 0, 20, 59, 69, 33, 69, 96, 73, 35, 40, 49, 50, 29, 46, 22, 0, 91, 86, 8, 95, 83, 95, 4, 89, 16, 81, 2]
    #y = [58, 46, 34, 7, 48, 25, 85, 78, 57, 86, 2, 81, 7, 62, 90, 3, 24, 61, 1, 27, 32, 56, 55, 46, 13, 89, 91, 42, 34, 21, 25, 12, 54, 26, 98, 11, 43, 99, 10, 53, 92, 78, 89, 97, 16, 74, 1, 13, 78, 63, 34, 69, 29, 15, 87, 61, 99, 27, 66, 98, 49, 3, 20, 46, 50, 18, 94, 2, 41, 80, 50, 33, 75, 27, 50, 46, 16, 65, 19, 83, 79, 92, 52, 19, 49, 9, 71, 2, 19, 83, 34, 12, 57, 49, 38, 37, 62, 86, 8, 33]

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
    #print(g)
    ##########ga
    
    #initiate
    population = ga.initialPopulationRandom(problemSize, populationSize)
    #print(population)
    
    startTime = time.time()
    #loop the GA
    #while bestFitness > 250:
    while generations < maxGen: #or (time.time()-startTime < 300):
        
        childPopulation = []
        fitnesses = []
        #get fitnesses for initial
        for chrom in population:
            fitnesses.append(ga.fitness(kNum, g, problemSize, chrom))
        #print(fitnesses)
        
        #find elites indexes and add elites to pool later
        eIndex1, eIndex2 = ga.elites(fitnesses, populationSize)
        elite1 = population[eIndex1]
        elite2 = population[eIndex2]
        #print(childPopulation)
        #print(ga.fitness(kNum, g, problemSize, childPopulation[0]))
        #print(ga.fitness(kNum, g, problemSize, childPopulation[1]))
        
        #SELECTION
        #while len(parentPopulation) < populationSize:
        #    parentPopulation.append(population[ga.tournament(fitnesses, populationSize)])
        parentPopulation = ga.roulette(fitnesses, populationSize, population)
        #print(parentPopulation)
        
        #fitnesses = []
        #get fitnesses for selected
        #for chrom in parentPopulation:
        #    fitnesses.append(ga.fitness(kNum, g, problemSize, chrom))
        
        #CROSSOVER
        while len(childPopulation) < populationSize:
            newChildren = ga.singlepoint_crossover(parentPopulation, populationSize, problemSize, crossoverRate)
            #newChildren = ga.vector_crossover(parentPopulation, populationSize, problemSize, crossoverRate)
            childPopulation.append(newChildren[0])
            childPopulation.append(newChildren[1])   
        #print(len(childPopulation))
        
        #fitnesses = []
        #get fitnesses for crossed over
        #for chrom in childPopulation:
        #    fitnesses.append(ga.fitness(kNum, g, problemSize, chrom))
        
        #MUTATION
        for child in childPopulation:
            child = ga.pairwise_exchange(child, problemSize, mutationRate)
        #    child = ga.nn_mutate(child, problemSize, mutationRate, g, kNum)
        seed = random.randint(0, populationSize-2)
        childPopulation[seed] = elite1
        childPopulation[seed+1] = elite2
        
        fitnesses = []
        #get fitnesses for mutated
        for chrom in childPopulation:
            fitnesses.append(ga.fitness(kNum, g, problemSize, chrom))
        
        #set child population to new parent population
        population = childPopulation
        
        #increment generation
        #generations = maxGen
        generations += 1
        if generations%(maxGen/10) == 0:
            bestFitness = fitnesses[ga.bestFitness(fitnesses)]
            print(bestFitness)
            print("--- %s seconds ---" % (time.time() - startTime))
            
    #output best fitness
    fitnesses = []
    for chrom in population:
        fitnesses.append(ga.fitness(kNum, g, problemSize, chrom))
    #print(fitnesses)
    bestFitness = fitnesses[ga.bestFitness(fitnesses)]
    print(population[ga.bestFitness(fitnesses)])
    print(bestFitness)
    #goodChrom = [0,5,10,15,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
    #print(goodChrom)
    #print(ga.fitness(kNum,g,problemSize, goodChrom))
    print("--- %s seconds ---" % (time.time() - startTime))
    print(generations)
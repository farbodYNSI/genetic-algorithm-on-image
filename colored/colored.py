import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

#----- should be assigned -----
maxIter = 30000
num_parents_mating = 3
crossover_perccent = 50
mutation_percent = 0.05
sol_per_pop = 9
saveFigures = False
#------------------------------

qualitiesR = []
qualitiesG = []
qualitiesB = []

imageWidth = 32
imageHeight = 32
pixelBits = 8

def img2vector(img_arr):
    fv = np.reshape(img_arr, -1)
    imgVec = []
    for pixel in fv:
        binaryValue = '{0:08b}'.format(pixel)
        for val in binaryValue:
            imgVec.append(int(val))
    imgVec = np.array(imgVec)
    return imgVec

def initialPopulation(nIndividuals=8):
    init_population = np.empty(shape=(nIndividuals,imageWidth*imageHeight*pixelBits),dtype=np.uint8)
    for indv_num in range(nIndividuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num, :] = np.random.randint(0, 2, imageWidth*imageHeight*pixelBits)
    return init_population

#fitness function
def fitness_fun(target_chrom, indiv_chrom):
    error=-1*np.sum(np.abs(indiv_chrom-target_chrom))
    return error

#fitnes values of all chrosomes
def calPopFitness(target_chrom, pop):
    qualities = np.zeros(pop.shape[0])
    for indiv_num in range(pop.shape[0]):
        qualities[indiv_num] = fitness_fun(target_chrom, pop[indiv_num, :])
    return qualities

def getBestParents(population, qualities,n=4):
    parents = []
    indx=np.argsort(qualities)[::-1] #should be sorted inverted because the higher the quality the better it is (sort is from lower to higher)
    n_fit=np.sort(qualities)[::-1]
    # take best n parents
    best_parent_indx=indx[0:n]
    for i in best_parent_indx:
        parents.append(population[i]) #n best parents from total population
    parents=np.array(parents)
    return parents

def crossover(parents, nIndividuals=8):
    #defining a blank array to hold the solutions after crossover
    new_population = np.empty(shape=(nIndividuals,imageWidth*imageHeight*pixelBits),dtype=np.uint8)
    #storing parents for the crossover operation
    new_population[0:parents.shape[0], :] = parents
    # Number offspring to be generated.
    num_newly_generated = nIndividuals-parents.shape[0]
    # All permutations for the parents selected
    parents_permutations = list(itertools.permutations(iterable=np.arange(0, parents.shape[0]), r=2))
    # Selecting some parents randomly from the permutations
    # به تعداد باقی مانده از پرنت ها رندوم از جایگشت ها برمیداریم
    selected_permutations = random.sample(range(len(parents_permutations)),num_newly_generated)
    comb_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        # Generating the offspring using the permutations previously selected randomly.
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]
        # Crossover by 1/50 th or 2 genes between 2 parents
        cross_size = np.int32(new_population.shape[1]*crossover_perccent/100)
        new_population[comb_idx+comb, 0:cross_size] = parents[selected_comb[0],0:cross_size]
        new_population[comb_idx+comb, cross_size:] = parents[selected_comb[1],cross_size:]
    return new_population

def mutation(population, numParentsMating, mutPercent):
    for indx in range(numParentsMating, population.shape[0]):
        # Selecting specific percentage of genes randomly
        rand_indx = np.uint32(np.random.random(size=np.uint32(mutPercent/100*population.shape[1]))*population.shape[1])
        # Genes are selected at a random an their values are changed
        new_values = np.random.randint(0, 2, len(rand_indx))
        # Updating the population .
        population[indx, rand_indx] = new_values
    return population

def showAndSave(iter , resultPopR,resultPopG,resultPopB,originalImg):
    #binary to pixel conversion
    resultList = []
    power = 7
    value = 0
    result = resultPopR[0]
    for i in range(len(result)):
        value += result[i]*(2**power)
        if power == 0:
            resultList.append(value)
            value=0
            power = 7
        else:
            power -= 1
    resultR = np.array(resultList)

    resultList = []
    power = 7
    value = 0
    result = resultPopG[0]
    for i in range(len(result)):
        value += result[i]*(2**power)
        if power == 0:
            resultList.append(value)
            value=0
            power = 7
        else:
            power -= 1
    resultG = np.array(resultList)

    resultList = []
    power = 7
    value = 0
    result = resultPopB[0]
    for i in range(len(result)):
        value += result[i]*(2**power)
        if power == 0:
            resultList.append(value)
            value=0
            power = 7
        else:
            power -= 1
    resultB = np.array(resultList)

    resultR = resultR.reshape((imageWidth,imageHeight))
    resultG = resultG.reshape((imageWidth,imageHeight))
    resultB = resultB.reshape((imageWidth,imageHeight))

    resultRGB = np.uint8(np.dstack((resultR,resultG,resultB)))
    #evaluation of reconstructed pixels
    evaluateR = np.ones_like(resultR)
    evaluateG = np.ones_like(resultG)
    evaluateB = np.ones_like(resultB)

    evaluateR[resultR != originalImg[:,:,0]]=0
    evaluateG[resultG != originalImg[:,:,1]]=0
    evaluateB[resultB != originalImg[:,:,2]]=0

    fig, axs = plt.subplots(2)
    axs[0].imshow(resultRGB)
    axs[1].imshow(originalImg)
    axs[0].text(evaluateR.shape[0],evaluateR.shape[1],'pixelsRatio = '+str(int(np.sum(evaluateR+evaluateG+evaluateB)/(evaluateR.shape[0]*evaluateR.shape[1]*3)*100)))
    figName = "colored"+str(iter)
    if saveFigures:
        plt.savefig(figName)
    plt.show()




image = cv2.imread('Untitled.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(imageWidth,imageHeight))
restoredPic=np.ones((imageWidth,imageHeight,3))


#storing image vector in a variable
imageVectorR=img2vector(image[:,:,0])
imageVectorG=img2vector(image[:,:,1])
imageVectorB=img2vector(image[:,:,2])

populationR=initialPopulation(sol_per_pop)
populationG=initialPopulation(sol_per_pop)
populationB=initialPopulation(sol_per_pop)

for iteration in range(maxIter):
    qualitiesR = calPopFitness(imageVectorR, populationR)
    qualitiesG = calPopFitness(imageVectorG, populationG)
    qualitiesB = calPopFitness(imageVectorB, populationB)

    print('Quality : ', qualitiesR, ', Iteration : ', iteration)

    parentsR = getBestParents(populationR, qualitiesR, num_parents_mating)
    parentsG = getBestParents(populationG, qualitiesG, num_parents_mating)
    parentsB = getBestParents(populationB, qualitiesB, num_parents_mating)

    populationR = crossover(parentsR, sol_per_pop)
    populationG = crossover(parentsG, sol_per_pop)
    populationB = crossover(parentsB, sol_per_pop)

    populationR = mutation(populationR, num_parents_mating, mutation_percent)
    populationG = mutation(populationG, num_parents_mating, mutation_percent)
    populationB = mutation(populationB, num_parents_mating, mutation_percent)


    if qualitiesR[0]==0 and qualitiesG[0]==0 and qualitiesB[0]==0:
        showAndSave(iteration,populationR,populationG,populationB,image)
        break

    if iteration == 100 or iteration == 1000 or iteration ==10000 or iteration ==20000:
        showAndSave(iteration,populationR,populationG,populationB,image)
else:
    showAndSave(maxIter,populationR,populationG,populationB,image)


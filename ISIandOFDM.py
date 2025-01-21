"""
PROBLEM STATEMENT:
Simulate 16-QAM (quadrature amplitude modulation) technique by implementing
the signal model y = x + z, where x belongs to 16-QAM with average energy unity,
and takes values with uniform distribution. Similar to the previous problem, z is a
Gaussian random variable with zero mean and variance N. For this setup,
a. Plot the average Symbol Error Probability as a function of the signal-to-noise-
ratio, defined as SNR = 10log10(1/N) from 0 dB to 20 dB in steps of 2 dB. Use
ML decoding method to obtain the plots.
b. Repeat the above step and plot the BER. Highlight the method used to map
the information bits to the symbols of the constellation.
c. Plot the Chernoff bound based expression for the Symbol Error Probability
and compare it with the above simulations-based plot on SER.
"""

#Solution begins here..
"""Since we are using 16 QAM, this implies there will be 16 points on the constellation space, provided that the average energy 
 of the constellation diagram is normalised to one."""

import math
import numpy as np
import matplotlib.pyplot as plt

def moveRight(position):
    newPosition = [position[0]+2, position[1]]
    return newPosition

def moveLeft(position):
    return [position[0]-2, position[1]]

def moveDown(position):
    return [position[0], position[1]-2]

def generateQAMRectangularSignalSpace(M):
    k = int(math.sqrt(M))
    currentPosition = [1-k , k-1]
    coordinates = []
    #coordinates.append(currentPosition)
    moveDownCounter = 0
    while moveDownCounter < k:
        coordinates.append(currentPosition)
        for i in range(k-1):
            if moveDownCounter%2 == 0:
                currentPosition = moveRight(currentPosition)
                coordinates.append(currentPosition)
            else: 
                currentPosition = moveLeft(currentPosition)
                coordinates.append(currentPosition)
        currentPosition = moveDown(currentPosition)
        moveDownCounter += 1
    return coordinates

def normaliseSignalSpace(coordinates):
    sumSquared = 0
    for i in range(len(coordinates)):
        sumSquared += coordinates[i][0]**2 + coordinates[i][1]**2
    averageEnergy = math.sqrt(sumSquared)/len(coordinates)
    normalisedCoordinates = []
    for i in range(len(coordinates)):
        normalisedCoordinates.append([coordinates[i][0]/averageEnergy,
                                      coordinates[i][1]/averageEnergy])
    return normalisedCoordinates

def createBlocksOfData(streamOfBits, M):
    #M is the number of elements that should be present in the group.
    if(len(streamOfBits)%M == 0):
        #no padding required, just proceed with the group already
        return [[streamOfBits[j] for j in range(i, i+M, 1)] for i in range(0, len(streamOfBits), M)]
    else:
        #padding required
        print("number of bits should be divisible by M")
        return

def encoderQAM(M, data):
    """
    inputs:
        M: number of constellation points on the signal space. 
        data: raw stream of information bits.
    Output:
        modulatedData: the data input to this function is converted to digitally modulated data and outputed
        mappings: a dictionary containing the information about how the data is mapped to constellation diagram
    """
    #first step: based on what the value of M is, we can generate the coordinates of constellation points. (basically our beta1, beta2)
    coordinates = generateQAMRectangularSignalSpace(M)
    normalisedQAM = normaliseSignalSpace(coordinates)

    #second step: based on what M is, we can determine how many bits will be represented by each constellation point on signal space
    m = int(math.log2(M)) #represents the number of bits represented by each constellation points, for this question it is 4
    blocksOfData = createBlocksOfData(data, m)

    possibleCombinations = [] #is the list containing all possible binary combinations of length m
    #the logic for filling all the possible combinations is written just below
    for i in range(M):
        x = bin(i)[2:]
        length = len(x)
        newX = (m-length)*'0'+x
        someList = []
        for j in range(m):
            someList.append(int(newX[j]))
        possibleCombinations.append(someList)
        del someList
    possibleCombinations.sort()
    print("all possible combinations are : ")
    print(possibleCombinations)
    #now the mapping of blocksOfData to the constellation begins, this is the step where the digital modulation begins
    mappings = {tuple(possibleCombinations[i]):tuple(normalisedQAM[i]) for i in range(len(possibleCombinations))} #converting the list type to tuple type because list type is unhashable in python
    #iterating over the blocks of data
    modulatedData = []
    for i in range(len(blocksOfData)):
        modulatedData.append(list(mappings[tuple(blocksOfData[i])]))
    return modulatedData, mappings

def MLEstimation(noisyData, constellationPoints):
    """
    inputs-->
        1. noisyData: this data needs to be estimated. This is a list type, containing sub lists of length = numberOfBasis
        2. constellationPoints: A list that contains all the constellation points coordinates
    output-->
        1. estimatedSymbols: a list of sublists(each having length=numberOfBasis)
    """
    estimatedSymbols = []
    for i in range(len(noisyData)):
        currentSymbol = noisyData[i]
        #compute the distance of current symbol with each of the symbols in the constellation points and find the minimum among them.
        d = 100000000 #unnecesarrily high value
        for j in range(len(constellationPoints)):
            norm = distance(currentSymbol, constellationPoints[j])  #distance function is defined just below to this function's definition
            if d > norm:
                iCap = j
                d = norm
        estimatedSymbols.append(constellationPoints[iCap])
    return estimatedSymbols

def distance(symbol1, symbol2):
    """
    inputs --> 
        1. symbol1: a list type having some elements of float type
        2. symbol2: another list having some elements of float type
        Constraint on input: they should have the same lengths and both should be of the type list
    Output:
        1. L2 norm/Euclidean distance between the symbols inputted
    """
    if len(symbol1) != len(symbol2):
        print("symbols should have the same length for L2 norm to be applicable")
        return None
    else:
        differenceList = [symbol1[i]-symbol2[i] for i in range(len(symbol1))]
        return np.linalg.norm(differenceList)
    
#here starts the main code:
#generate uniformly distributed data:
numberOfBits = 800
numberOfBasis = 2

data = np.random.choice([0,1], size=numberOfBits)

#need to map this data into the signal space
#to do this, first we need to generate the normalised signal space
M = 16

encoderOutput = encoderQAM(M, data)
QAMModulatedData = encoderOutput[0]
mappings = encoderOutput[1]

#adding noise to the QAMModulatedData
SNRdB = [2*i for i in range(11)]
variances = [math.pow(10, -SNRdB[i]/10) for i in range(len(SNRdB))]

#generating noise correponding to each variance value
noises = [np.random.normal(0, math.sqrt(variances[i]), 
                           size = len(QAMModulatedData)*len(QAMModulatedData[0]))
          for i in range(len(variances))]
SERS = [] #will collect the symbol error rate in each iteration of noise variance
#analysing each case of noise
for i in range(len(noises)):
    blocksOfNoise = createBlocksOfData(noises[i], numberOfBasis)
    noisyData = [[QAMModulatedData[p][0]+blocksOfNoise[p][0],
                  QAMModulatedData[p][1]+blocksOfNoise[p][1]] 
                  for p in range(len(QAMModulatedData))]

    #just for the heck of it, let's see how the noisy data looks like on the scatter plot
    """
    if i == len(noises)-3:
        plt.scatter([noisyData[p][0] for p in range(len(noisyData))] , [noisyData[p][1] for p in range(len(noisyData))])
        plt.show()
    """
    #let's finally decode this noisy Data
    constellationPoints = normaliseSignalSpace(generateQAMRectangularSignalSpace(M))
    estimatedSymbols = MLEstimation(noisyData, constellationPoints)
    #next step is to capture the number of symbols in error.
    symbolsInError = 0
    for i in range(len(estimatedSymbols)):
        if estimatedSymbols[i] != QAMModulatedData[i]:
            symbolsInError+=1
    SERS.append(symbolsInError/len(estimatedSymbols))
plt.plot(SNRdB, SERS)
plt.title("ML decoding of 16QAM scenario")
plt.xlabel("SNR in dB")
plt.ylabel("Symbol Error Rate")
plt.show()
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import requests

def modulationBPSK(data): #input to this is the list of data bits
    modulatedData = []
    for i in range(len(data)):
        modulatedData.append((2*data[i]-1)/math.sqrt(2)) #dividing by sqrt(2) ensures that the signal space is normalised to uniform energy
    return modulatedData

def generateQPSKSignalSpace():
    #returns a dictionary
    return {(0,0): (1+1j)/math.sqrt(2),
            (0,1): (-1+1j)/math.sqrt(2),
            (1,0): (-1-1j)/math.sqrt(2),
            (1,1): (1-1j)/math.sqrt(2)}

def modulationQPSK(data):
    #NOTE: QPSK has the tendency to carry information about bits per symbol, so we will need to club the databits into tuples of two.
    modulatedData = []
    #let the mapping to the QPSK symbols be as follows:
    """
    {(0,0): (1+1j)/math.sqrt(2),
     (0,1): (-1+1j)/math.sqrt(2),
     (1,0): (-1-1j)/math.sqrt(2),
     (1,1): (1-1j)/math.sqrt(2)}
    """
    #clubbing the data bits into tuples of two:
    dataClubbed = []
    ctr = 0
    while ctr < len(data):
        dataClubbed.append((data[ctr], data[ctr+1]))
        ctr += 2
    #now starts the mapping
    qpskSignalSpace = generateQPSKSignalSpace()
    #normalising the signal space to unit energy
    for key in qpskSignalSpace:
        qpskSignalSpace[key] = qpskSignalSpace[key]/2   #total energy of the qpsk signal space is 4 units, so we usually divide by sqrt(total energy)
    modulatedData = []
    for i in range(len(dataClubbed)):
        modulatedData.append(qpskSignalSpace[dataClubbed[i]])
    return modulatedData

def generate_cn_noise(N):  #generates CN(0,1) noise of length N
    real_part = np.random.normal(0, np.sqrt(0.1), N)  # Real part: N(0, 1)
    imag_part = np.random.normal(0, np.sqrt(0.1), N)  # Imag part: N(0, 1)
    noise = real_part + 1j * imag_part                # Combine into CN(0, 1)
    return noise

def addCyclicPrefix(data, L):
    data = data.tolist()
    if L <= len(data):
        #this condition will only make sense:
        prefixList = data[-L:-1]
        prefixList.append(data[-1])
        return prefixList+data
    else:
        print("ERROR, cannot add cyclic prefix for the given length L")

def MLDecoderBPSK(symbol):
    if abs(symbol-1) <= abs(symbol+1):
        return 1
    else:
        return -1
    
#fetching image and it's data
pathToImage = "https://media.geeksforgeeks.org/wp-content/uploads/20230717130922/Ganeshji.webp"
response = requests.get(pathToImage)
if response.status_code == 200:
        # Load the image and convert to grayscale
    image = Image.open(BytesIO(response.content)).convert("L")
    original_shape = np.array(image).shape  # Save original shape
    print(f"Original Image Shape: {original_shape}")
    
    # Convert grayscale image to binary data (0 and 1)
    bits = np.array(image)
    binary_data = (bits > 127).astype(int)  # Thresholding at 127 to get binary values (0 or 1)
    # Flatten the binary array into a 1D array
    data = binary_data.flatten()
    print(f"Flattened Data Length: {len(data)}")
    #now let us define number of carries and PDP list
    noOfCarriers = [16,32,64]
    PDP = [[1, 0.3], [1,0,0,0.3], [1,0.2,0.1]]
    BERBPSK = []
    #here starts the main portion of the code:
    for i in range(len(PDP)):
        currentPDP = PDP[i]
        for j in range(len(noOfCarriers)):
            N = noOfCarriers[j] #defines the value of N, which will further be used to compute the N point IDFT and DFT respectively.
            #next we generate the channel taps [h0, h1, h2, ...], this can be done using the formula: h[i]=sqrt(currentPDP[i])*g, here g is CN(0, 1)
            channelTaps = []   #h
            for k in range(len(currentPDP)):
                #g = (np.random.normal() + 1j*np.random.normal())/math.sqrt(2) #the scaling with sqrt(2) is done in order to ensure that g has a variance = 1
                g = 1
                channelTaps.append(g*math.sqrt(currentPDP[k]))
            [0 for i in range(N-len(channelTaps))].extend(channelTaps)
            #H
            H = np.fft.fft(channelTaps)
            lambdaMatrix = [[0 for rows in range(N)] for cols in range(N)]
            """
            for k in range(N):
                lambdaMatrix[k][k] += H[k]"""
            #case 1: BPSK
            BPSKModulatedData = modulationBPSK(data)
            #need to partition the modulated symbols into groups of N each
            #zero padding the data
            noOfZeroes = N-len(BPSKModulatedData)%N
            for k in range(noOfZeroes):
                BPSKModulatedData.append(0)
            IDFTBPSKModulatedData = []
            corruptedData = []
            for k in range(0, len(BPSKModulatedData), N):
                currentChunk = BPSKModulatedData[k:k+N]
                #take the N point idft of this chunk
                x = np.fft.ifft(currentChunk)*math.sqrt(N) #multiplying by sqrt(N) to account for U hermitian being unitary.
                """Adding cyclic prefix here : """
                xWithCyclicPrefix = addCyclicPrefix(x , len(channelTaps))
                h_convolve_xWithCyclicPrefix = np.convolve(channelTaps , xWithCyclicPrefix)
                #corrupt this above signal with CN(0,1)
                noise = generate_cn_noise(len(h_convolve_xWithCyclicPrefix))
                corruptedChunk = []
                for l in range(len(noise)):
                    #corruptedChunk.append(noise[l]+h_convolve_xWithCyclicPrefix[l])
                    corruptedChunk.append(h_convolve_xWithCyclicPrefix[l])
                corruptedData.append(corruptedChunk)
            #the corruptedData is the actual data that is the output of the decorrelator at the receiver
            estimatedDigitalData = []
            for k in range(len(corruptedData)):
                currentChunk = corruptedData[k]
                #remove CP from this chunk
                currentChunkMinusCP = currentChunk[len(channelTaps):]
                #take the fft
                fft_currentChunkMinusCP = np.fft.fft(currentChunkMinusCP)/math.sqrt(N)
                #decode using ML decoding
                for l in range(len(fft_currentChunkMinusCP)):
                    estimatedDigitalData.append(MLDecoderBPSK(fft_currentChunkMinusCP[l]))
            estimatedBits = []
            for k in range(len(estimatedDigitalData)):
                #demodulate the estimatedDigitalData into bits 0 and 1
                if estimatedDigitalData[k] < 0:
                    #map this to bit 0
                    estimatedBits.append(0)
                else:
                    estimatedBits.append(1)
            #computing the BER
            BitsInError = 0
            for k in range(min(len(estimatedDigitalData), len(data))):
                if estimatedBits[k] == data[k]:
                    pass
                else:
                    BitsInError+=1
            BERBPSK.append(BitsInError)
            #reconstructing the image using this data to analyse the result.

                
            #case 2: QPSK
            QPSKModulatedData = modulationQPSK(data)
            noOfZeroes = N-len(QPSKModulatedData)%N
            for k in range(noOfZeroes):
                QPSKModulatedData.append(0+0j)
            
else:
    print("Cannot load the image from the internet.. make sure you have the internet active")
print(len(data))
print(BERBPSK)
"""1. Simulate the BPSK modulation technique by implementing the signal model y = x + z,
where x belongs to the set {-1,1} with uniform distribution, and z is a Gaussian
random variable with zero mean and variance N. For this setup,
a. Plot the average Bit Error Probability as a function of the signal-to-noise-ratio,
defined as SNR = 10log10(1/N) from 0 dB to 20 dB in steps of 2 dB. Use ML
decoding method to obtain the plots
b. Repeat the above step and use an appropriate decoding method when the
BPSK symbols take values {1, -1} with probability {p, 1-p}, for some 0< p < 1,
respectively. You may plot multiple BER curves for p = 0.2, 0.4, 0.6, 0.8. What
is your inference on the behavior of the plots as a function of p and SNR.
c. Plot the Chernoff bound based expression for the BER and compare it with
the above plots."""

import math
import matplotlib.pyplot as plt
import numpy as np

#Since we are going to use a BPSK, this implies we will be using effectively only one basis. 
#Although I think we can use multiple basis as well as long as the phase shift between the selected constellation points is 180 degrees.
#But using multiple basis would introduce redundancy, as the data will not carry any additional information, and moreover multiple basis implies use of multiple oscillators at transmitter and receiver.
#Hence it is conclusive from the above statements that using multiple basis is trivial as long as we can live with a single basis.

def MLDecoderBPSK(symbol):
    if abs(symbol-1) <= abs(symbol+1):
        return 1
    else:
        return -1

numberOfBits = 1000
#generating data
data = np.random.choice([0,1], size=numberOfBits)
x = []#modulated data, mapping bit 1 to symbol 1, and bit 0 to symbol -1
for i in range(len(data)):
    if data[i]==0:
        x.append(-1)
    else:
        x.append(1)

#now we need to generate noise based on SNR values given:
SNRdB = [2*(i+1) for i in range(11)]
variances = [10**(-SNRdB[i]/10) for i in range(len(SNRdB))] #this gives N in linear scale.

#generating noises corresponding to these variances:
noises = [np.random.normal(0, math.sqrt(variances[i]), size = numberOfBits) for i in range(len(variances))]
BERSML = []
for i in range(len(noises)):
    noisyData = [x[k]+noises[i][k] for k in range(numberOfBits)]
    decodedData = [MLDecoderBPSK(noisyData[i]) for i in range(len(noisyData))]
    numberOfBitsInError = 0
    for j in range(len(decodedData)):
        if decodedData[j] != x[j]:
            numberOfBitsInError += 1
    BERSML.append(numberOfBitsInError/numberOfBits)
    del noisyData, decodedData

#average probability of error: Q(1/sqrt(N) in each case, N denotes the noise variance
#OR, instead of taking Q(1/(N)), let's take into account the entire SNR instead. So instead of plotting Q(1/N), let's plot Q(sqrt(SNR)), here SNR is taken in linear scale and is equal to 1/N
#Q(x)=0.5-0.5erf(x/sqrt(2))
SNRLinear = [10**(SNRdB[i]/10) for i in range(len(SNRdB))]
averageProbabilityOfError = [0.5*math.erfc(math.sqrt(SNRLinear[i])/math.sqrt(2)) for i in range(len(SNRLinear))]

#graphplotting
plt.subplot(2,2,1)
plt.title("ML decoding")
plt.plot(SNRdB, BERSML)
plt.xlabel("SNR in dB")
plt.ylabel("Bit Error Rate")

#graph plotting for average probability of error.
plt.subplot(2,2,2)
plt.title("Average probabiltiy of error in the ML case")
plt.xlabel("SNR in dB")
plt.ylabel("Average probability of Error")
plt.plot(SNRdB, averageProbabilityOfError)

# b) part, the symbols are not equiprobable. 
def MAPDecoderBPSK(symbol, probability):
    #assuming given is the probability of the symbol 1 in the function signature.
    #output of this decoding is based on arg_max(exp(-()))
    if math.exp(-(symbol-1)**2)*probability >= math.exp(-(symbol+1)**2)*(1-probability):
        return 1
    else:
        return -1

probabilities = [0.2*i for i in range(1,5)]
BERMAP = []

for t in range(len(probabilities)):
    for i in range(len(noises)):
        noisyData = [x[k]+noises[i][k] for k in range(numberOfBits)]
        decodedData = [MAPDecoderBPSK(noisyData[i], probabilities[t]) for i in range(len(noisyData))]
        numberOfBitsInError = 0
        for j in range(len(decodedData)):
            if decodedData[j] != x[j]:
                numberOfBitsInError += 1
        BERMAP.append(numberOfBitsInError/numberOfBits)
        del noisyData, decodedData

    #graph plotting
    plt.subplot(2,2,3)
    plt.title("MAP decoding")
    plt.plot(SNRdB, BERMAP, label = "p="+str(probabilities[t]))
    plt.xlabel("SNR in dB")
    plt.ylabel("BER")
    plt.legend()
    BERMAP = []

    #plotting average probability of error:
    

#part c:
#plot of chernoff bound. 

dmin_overall = abs(1-(-1))
chernoffBoundML = [2*math.exp(-(dmin_overall**2)/(4*variances[i])) for i in range(len(variances))]
plt.subplot(2,2,2)
plt.plot(SNRdB, chernoffBoundML, label="chernoff bound")
plt.xlabel("SNR in dB")
plt.ylabel("Upper bound on Probability of Error")
plt.legend()


#chernoff bound for the MAP case is given by:
"""
Pe <= p * exp{-[ln(p / (1-p)] * (1/32*No)}"""
chernoffs = []
for p in probabilities:
    bound = []
    for var in variances:
        bound.append(p*math.exp(-(math.log(abs(p/(1-p)))**2 / (8*var)))
                     +(1-p)*math.exp(-(math.log(abs((1-p)/p)**2) / (8*var))))
    chernoffs.append(bound)
ctr = 1
for bounds in chernoffs:
    plt.subplot(2,2,3)
    plt.plot(SNRdB, bounds, label = "chernoff for p = " + str(0.2*ctr))
    ctr += 1
plt.show()


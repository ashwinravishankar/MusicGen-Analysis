#Import the necessary library files
%matplotlib inline 
import IPython.display as MusicDisplay
import sklearn
import librosa
import numpy
import scipy
import matplotlib.pyplot as plt
import librosa.display as LibDisplay
import mir_eval

# Convert the WAV file to floating point time series - monophony
# Sampling Frequency - Number of samples per second.
timeSeries, samplingRate = librosa.core.load('/Users/ashwin/GeorgeMasonUniversity/CS688/Generated/output_audio.wav', sr=22050, duration=30 )
# print(samplingRate)
frameCount = librosa.util.frame(y=timeSeries)
print(frameCount.shape)
print(timeSeries.shape)
# Plot the time series of audio in wave form
plt.figure(figsize=(16, 5)) 
LibDisplay.waveplot(timeSeries, samplingRate)

# Detect onset sequences in the time series - get array of frames
#   delta - threshold offset for mean
#   wait - number of samples to wait after picking a peak
onsetFrames = librosa.onset.onset_detect(y=timeSeries, sr=samplingRate, hop_length=512)
# print(onsetFrames)

# Converts frame counts to time (seconds).
onsetTimes = librosa.core.frames_to_time(frames=onsetFrames, sr=samplingRate)
# print(onsetTimes)

plt.figure(figsize=(16, 5)) 
LibDisplay.waveplot(timeSeries, samplingRate)
plt.vlines(onsetTimes, ymin=-0.4, ymax=0.4, color='r')

# Converts frame indices to audio sample indices.
onsetSamples = librosa.core.frames_to_samples(frames=onsetFrames)
print(onsetSamples)

## FEATURES - ANALYSING AND PLOTTING FEATURES

# Zero Crossing Rate
#   The number of times the signal crosses the horizontal axis.
frame1 = 320000
frame2 = 320145
plt.figure(figsize=(16, 5))
plt.plot(timeSeries[frame1:frame2])
plt.hlines(0, 0, 145, color='r')

# Total zero crossing count
zeroCrossCount = librosa.core.zero_crossings(timeSeries).sum()
print(zeroCrossCount)

# Mel-frequency cepstral coefficients (MFCCs)
mfccs = librosa.feature.mfcc(y=timeSeries, sr=samplingRate, n_mfcc=20)

# Plot MFCC
plt.figure(figsize=(17, 5))
LibDisplay.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.tight_layout()

# Root Mean Square Energy (RSME)
frame_length=2048
hop_length=512
rmse = librosa.feature.rmse(y=timeSeries, frame_length=frame_length, hop_length=hop_length)

print(rmse.shape)
rmse=rmse[0]

# Signal Energy
energy = numpy.array([
    sum(abs(timeSeries[i:i+frame_length]**2)) 
    for i in range(0, len(timeSeries), hop_length)
])

print(energy.shape)

# Plot Signal Energy and RMSE.
frameCount = range(len(energy))
t = librosa.core.frames_to_time(frameCount, sr=samplingRate, hop_length=hop_length)

plt.figure(figsize=(17, 5))
LibDisplay.waveplot(timeSeries, sr=samplingRate, alpha=0.4)
plt.plot(t, energy/energy.max(), 'r--')
plt.plot(t[:len(rmse)], rmse/rmse.max(), color='g')
plt.legend(('Energy', 'RMSE'))

## FEATURE EXTRACTION - COMPUTE 2 FEATURES

def LearnFeatures(samples):
    firstFeature = librosa.core.zero_crossings(samples).sum()
    secondFeature = scipy.linalg.norm(samples)
    return [firstFeature, secondFeature]

# Features that can be used for plotting

# Signal Energy (compute vector norm)
#     scipy.linalg.norm(samples)

# Zero Crossing Rate
#     librosa.core.zero_crossings(samples).sum()

# RMSE
#     librosa.feature.rmse(y=timeSeries, frame_length=frame_length, hop_length=hop_length)

# MFCCs
#     librosa.feature.mfcc(y=timeSeries, sr=22050, S=None, n_mfcc=20)

# Spectral Centroid
#     librosa.feature.spectral_centroid(y=timeSeries, sr=22050)


# Chroma
#     librosa.feature.chroma_stft(y=timeSeries, sr=22050) 

frameSize = int(samplingRate * 0.1)
featuresArray = numpy.array([
    LearnFeatures(timeSeries[i : i+frameSize])
    for i in onsetSamples
])
# print(featuresArray)

# Scatter plot of the features before being clustered
plt.figure(figsize=(17, 5))
plt.scatter(featuresArray[:,0], featuresArray[:,1])

## SCALING THE FEATURES BEFORE BEING PLOTTED [-1, 1]
scalingComponent = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

featuresArray = scalingComponent.fit_transform(featuresArray)

# Plot scaled features
# plt.figure(figsize=(17, 5))
plt.scatter(featuresArray[:,0], featuresArray[:,1])

## CLUSTERING ALGORITHMS FROM SCIKIT-LEARN

# K-Means Clustering Algorithm 
kmeans = sklearn.cluster.KMeans(n_clusters=2)
labelsKMeans = kmeans.fit_predict(featuresArray)
print(labelsKMeans)

# Plot K-Means
plt.scatter(featuresArray[labelsKMeans==0, 0], featuresArray[labelsKMeans==0, 1], c='b')
plt.scatter(featuresArray[labelsKMeans==1, 0], featuresArray[labelsKMeans==1, 1], c='r')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Signal Energy')
plt.legend(('Class A', 'Class B'))

centersKMeans = kmeans.cluster_centers_
plt.scatter(centersKMeans[:, 0], centersKMeans[:, 1], c='black', s=200)

# Affinity Propogation Clustering Algorithm 
affinityPropogation = sklearn.cluster.AffinityPropagation()
labelsAffinityPropogation = affinityPropogation.fit_predict(featuresArray)
print(labelsAffinityPropogation)

plt.scatter(featuresArray[:, 0], featuresArray[:,1], c=labelsAffinityPropogation, cmap='viridis')
plt.legend((labelsAffinityPropogation))

centersAffinityPropogation = affinityPropogation.cluster_centers_
plt.scatter(centersAffinityPropogation[:, 0], centersAffinityPropogation[:, 1], c='black', s=200)

# DBSCAN Clustering Algorithm 
dbscanModel = sklearn.cluster.DBSCAN()
labelsdbscan = dbscanModel.fit_predict(featuresArray)
print(labelsdbscan)

plt.scatter(featuresArray[:, 0], featuresArray[:,1], c=labelsdbscan, cmap='viridis')
plt.legend((labelsdbscan))


## Week 2
### Planning
- Get familiar with Torcs by playing with my_driver.py
- Implement a very simple regression neural network with targets acceleration, brake, and steer 
- Use autoencoders to reduce the feature space of the sensor data

### Output
- A much better understanding of the torcs environment and RNNs
- Feed forward neural network that predicts same values for every timestep
- Recurrent neural network that almost gives some output but has some dimention issue (easy to fix)
- Autoencoder (does it work Alex?)
- Able to drive the car ourselves, but not yet able to gather sensor data

### Questions
- How to collect sensor data from manually driving the car?
- How do we implement an Echo State Network? There aren't any good looking packages that we can find.

## Week 3
### Planning 
- Create LSTM
- Understand training data


### Output
- Produced training data based on rule based driving
- Gained insight into driver calls (i.e. when is it called?)
- LSTM working
- Sent e-mail to Mark (call for help)

### Questions
- All versions of NN are not working on the data. Why is this? Maybe (3-target) regression is not the best solution?
- Fundamental wrong approach (wait for e-mail Mark)?
- Error in LSTM?
- Reinforcement Learning?

## Week 4
### Planning
- Feature whitening
- Feature rescaling
- Regularization (dropout)
- Tanh and softmax output layer in two different networks (This forces the network to have an output in the desired range)
- Skeleton for NN part of report 
- Cap acc, break and steer manually
- Use PCA to reduce feature space (might help because sensors have a lot of overlap so this might reduce the amount of noice present in the features. As we are using a relatively small neural network, a smaller feature space could improve results)


### NEAT Phase 2: 
- Use simple bot to generate initial steer, acc and brake commands
- One individual in the population is one (initially simple) network with as input all sensors, the opponent sensors and the steer, acc and brake. Output is the finetuned steer, acc and brake. 
- The fitness function will be distance raced, cars overtaken, how much damage incurred and how long it's been off track.
- Every 20 seconds of the race another individual will be tested and its fitness summed up. 
- Before every individual switch, the car must be put in the middle of the track if it's been left off track.
- 

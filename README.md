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
- Switch back to default bot when edge track sensors give value of -1 (off track)
- Switch to simple full throttle bot when front sensors see no edge 

### Online-line NEAT Phase 2:
- Initialize a population of 100 individuals (simple FFNN without hidden layer and with 3 output nodes
- Start a race with 100 laps.
- Let individual i drive after bot warm-up, and save distance raced, cars overtaken/overtaken by, damage, time off track.
- After a time (30 seconds), remove individual i from the steering wheel, use 5 second warmup time with simple bot, and switch to individual i+1
- After all individuals have had their time (pauze game?), evaluate, select and mutate/recombine.
- Test new population

### Combine Phase 1 and Phase 2:
a. Use the last hidden layer of Phase 1 as input to the to be evolved network from Phase 2 instead of regular sensorinput, along with the added (new) opponents data so it learns how to handle opponents.
b.  Use all available sensor data and evolve from scratch.
c.

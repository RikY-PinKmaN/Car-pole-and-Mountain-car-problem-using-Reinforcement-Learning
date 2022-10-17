clear all
% Fix the random generator seed for reproducibility
rng(1,'twister');
%% cart pole with with state and action inputs
% load predefined environment
env = rlPredefinedEnv("CartPole-Discrete");
% get observation and specification info
stateInfo = getObservationInfo(env); % get state info for cart pole demo
stateDimension = stateInfo.Dimension; % get continuous state dimension
actionInfo = getActionInfo(env); % get action info for cart pole demo
numActions = length(actionInfo.Elements); % number of discrete actions
%% Define a Q-function network with state-only input (known in Matlab as a critic)
% Create a shallow neural network to approximate the Q-value function
net = [imageInputLayer(stateDimension,'Normalization','none','Name','myobs')
fullyConnectedLayer(10)
reluLayer
fullyConnectedLayer(numActions)];
% Create the critic using the network
critic = rlQValueRepresentation(net,stateInfo,actionInfo,'Observation',{'myobs'})
% To check your critic, use the getValue function to return a random observation
v = getValue(critic,{rand(stateDimension)})
%% Specify agent options, and create an agent using the environment and critic definition
% agent options
agentOpts = rlDQNAgentOptions(...
'UseDoubleDQN',false, ...
'TargetUpdateMethod',"periodic", ...
'TargetUpdateFrequency',100, ...
'ExperienceBufferLength',100000, ...
'DiscountFactor',0.99, ...
'MiniBatchSize',256);
% define the agent
agent = rlDQNAgent(critic,agentOpts)
%% Train the agent
% specify reinforcement learning training options
trainOpts = rlTrainingOptions(...
'MaxEpisodes', 100, ...
'MaxStepsPerEpisode', 300, ...
'Verbose', false, ...
'Plots','training-progress',...
'StopTrainingCriteria','AverageReward',...
'StopTrainingValue',300);
% train the agent
trainingStats = train(agent,env,trainOpts);
%% Simulate
% Fix the random generator seed for reproducibility
rng(1,'twister');
plot(env) % visualise the cart-pole system
simOptions = rlSimulationOptions('MaxSteps',300); % simulation options
experience = sim(env,agent,simOptions); % log experience
totalReward = sum(experience.Reward) % extract reward
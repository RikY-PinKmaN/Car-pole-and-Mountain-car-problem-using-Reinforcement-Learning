% Reinforcement learning for mountain car
clear all
% define the random number seed for repeatable results
rng(1,'twister');
% environment definition for mountain car
% observation info
ObservationInfo = rlNumericSpec([2 1]);
ObservationInfo.Name = 'States';
ObservationInfo.Description = 'Position and Velocity';
% action info
ActionInfo = rlFiniteSetSpec([-1 0 1]);
ActionInfo.Name = 'Action';
% environment
env = rlFunctionEnv(ObservationInfo,ActionInfo,'stepDynamics','initialDynamics');
% get observation and specification info
stateInfo = getObservationInfo(env); % get state info
stateDimension = stateInfo.Dimension; % get continuous state dimension
actionInfo = getActionInfo(env); % get action info for cart pole demo
numActions = length(actionInfo.Elements); % number of discrete actions
% Define a Q-function network with state-only input (known in Matlab as a critic)
% Create a shallow neural network to approximate the Q-value function
net = [imageInputLayer(stateDimension,'Normalization','none','Name','States')
fullyConnectedLayer(24)
reluLayer
fullyConnectedLayer(numActions)];
% Create the critic using the network
critic = rlQValueRepresentation(net,stateInfo,actionInfo,'Observation',{'States'});
% To check your critic, use getValue to return the values of a random input
v = getValue(critic,{rand(stateDimension)})
% Define agent and train
% agent options
agentOpts = rlDQNAgentOptions(...
'UseDoubleDQN',false, ...
'TargetUpdateMethod',"periodic", ...
'TargetUpdateFrequency',4, ...
'ExperienceBufferLength',100000, ...
'DiscountFactor',0.99, ...
'MiniBatchSize',256);
% define the agent
agent = rlDQNAgent(critic,agentOpts)
% specify reinforcement learning training options
trainOpts = rlTrainingOptions(...
'MaxEpisodes', 100, ...
'MaxStepsPerEpisode', 1000, ...
'Verbose', false, ...
'Plots','training-progress',...
'StopTrainingCriteria','AverageReward');
% train the agent
trainingStats = train(agent,env,trainOpts);
%% Simulate
simOptions = rlSimulationOptions('MaxSteps',1000); % simulation options
experience = sim(env,agent,simOptions); % log experience
totalReward = sum(experience.Reward); % extract reward
% plot simulation results
figure;
% plot training rewards
subplot(2,1,1);
plot(trainingStats.EpisodeIndex,trainingStats.EpisodeReward);
hold on;
plot(trainingStats.EpisodeIndex,trainingStats.AverageReward);
title('Training Rewards');
xlabel('Episode');
ylabel('Reward');
legend('Episode Reward','Average Reward','Location','SouthEast');
nt = length(experience.Reward.Time);
States = experience.Observation.States.Data;
xvals = [-1.2:0.1:0.6];
yvals = sin(3*xvals);
cumReward = cumsum(experience.Reward.Data);
% plot animation
for i = 1:nt
subplot(2,1,2); % new subplot
cla; % clear plot
plot(xvals,yvals,'k'); % plot mountain
hold on
x = States(1,1,i); % horizontal position of mountain car
y = sin(3*x); % height of mountain car
plot(x,y,'sq','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
title(['Mountain Car Animation Reward = ' num2str(cumReward(i))])
pause(0.01)
end
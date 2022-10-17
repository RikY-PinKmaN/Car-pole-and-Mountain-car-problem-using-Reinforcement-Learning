%% Tabular reinforcement learning for cart pole control problem
% the task is to balance a pole on a cart!
clear all
%% setup the problem: cart pole specific parameters
% first define the random number seed for repeatable results
rng(1,'twister');
% number of (quantized) states and actions
nquant = 100; % assume 100 quantization steps for each state
nstates = 3; % 3 states needed: xdot, theta, thetadot
ns = nquant^nstates; % number of discrete states
na = 2; % number of actions: 2 actions, -F or +F
% define scaling factors for quantizing the continuous states
xdotmax = 10; xdotmin = -10; % xdot min/max
thetamax = 30 * pi/180; thetamin = -30 * pi/180; % theta min/max
thetadotmax = 2*pi; thetadotmin = -2*pi; % thetadot min/max
delta = [(xdotmax-xdotmin) (thetamax-thetamin) ...
    (thetadotmax-thetadotmin)]'./(nquant-1); % min/max ranges
fmin = [xdotmin thetamin thetadotmin]'; % state min vals
% pack parameters into an auxiliary variable for function passing
Aux.nquant = nquant; % number of quantization steps
Aux.delta = delta; % min/max state ranges
Aux.fmin = fmin; % min state values
% action space
action = [-10 10];
%% reinforcement learning parameters
alpha = 0.9; % learning rate
epsilon = 0.3; % exploration probability for epsilon greedy
gamma = 0.2; % discount factor
N = 2e4; % number of learning episodes
T = 200; % number of time steps
Q = zeros(ns,na); % initialise Q table
% loop over episodes
avgR = 0; % initialise moving average reward
h1 = figure; clf; hold on; % initialise plot
for n = 1:N
    % initialise episode
    State = initialState; % initialise the state randomly or to zero
    next_j = stateIdx(State,Aux); % extract state index
    R = 0; % initialise cumulative reward
    % loop over time for one episode
    for t = 1:T
        % update state index j
        j = next_j;
        % define the action using epsilon greedy
        if rand < epsilon
            i = randi(2); % generate index for random action
        else
            [~,i] = max(Q(j,:)); % generate index for optimal action
        end
        A = action(i);
        % update state using simulation of the environment
        [State,Reward,flag] = mySimulation(State,A);
        % break if termination condition met
        if flag == 1
            break
        end
        % define next state index
        next_j = stateIdx(State,Aux); % extract state index
        % Q-learning update
        Q(j,i) = Q(j,i) + alpha*(Reward + gamma*max(Q(next_j,:)) - Q(j,i));
        % cumulative reward
        R = R + Reward;
    end
    % avg reward
    avgR = 0.99*avgR + 0.01*R;
    % plot episode reward
    if mod(n,100) == 0
        figure(h1);
        plot(n,R,'k.');
        plot(n,avgR,'.r','markersize',10);
        xlim([0 N]); ylim([0 200]);
        xlabel('Episode'); ylabel('Reward');
        title('Convergence Monitoring: Reward and Average Reward');
        legend('Episode Reward','Average Reward');
        drawnow;
    end
end
% initialise episode
T = 200; % loop time for evaluation
State = initialState; % initialise the state randomly or to zero
next_j = stateIdx(State,Aux); % extract state index
% Initiate the visualization
h2 = figure('Visible','on','HandleVisibility','off');
ha = gca(h2); ha.XLimMode = 'manual'; ha.YLimMode = 'manual';
ha.XLim = [-3 3]; ha.YLim = [-1 2];
hold(ha,'on');
% loop over time-steps
for t = 1:T
    % update j
    j = next_j;
    % policy: define the action using greedy method
    [~,i] = max(Q(j,:));
    A = action(i);
    % update state
    [State,Reward,flag] = mySimulation(State,A);
    % discretise the continuous state using quantization (ignore X)
    next_j = stateIdx(State,Aux); % extract state index
    % plot
    % Set visualization figure as the current figure
    ha = gca(h2);
    % Extract the cart position and pole angle
    x = State(1);
    theta = State(3);
    cartplot = findobj(ha,'Tag','cartplot');
    poleplot = findobj(ha,'Tag','poleplot');
    if isempty(cartplot) || ~isvalid(cartplot) ...
            || isempty(poleplot) || ~isvalid(poleplot)
        % Initialize the cart plot
        cartpoly = polyshape([-0.25 -0.25 0.25 0.25],[-0.125 0.125 0.125 -0.125]);
        cartpoly = translate(cartpoly,[x 0]);
        cartplot = plot(ha,cartpoly,'FaceColor',[0.8500 0.3250 0.0980]);
        cartplot.Tag = 'cartplot';
        % Initialize the pole plot
        HalfPoleLength = 0.5;
        L = HalfPoleLength*2;
        polepoly = polyshape([-0.1 -0.1 0.1 0.1],[0 L L 0]);
        polepoly = translate(polepoly,[x,0]);
        polepoly = rotate(polepoly,rad2deg(theta),[x,0]);
        poleplot = plot(ha,polepoly,'FaceColor',[0 0.4470 0.7410]);
        poleplot.Tag = 'poleplot';
    else
        cartpoly = cartplot.Shape;
        polepoly = poleplot.Shape;
    end
    % Compute the new cart and pole position
    [cartposx,~] = centroid(cartpoly);
    [poleposx,poleposy] = centroid(polepoly);
    dx = x - cartposx;
    dtheta = theta - atan2(cartposx-poleposx,poleposy-0.25/2);
    cartpoly = translate(cartpoly,[dx,0]);
    polepoly = translate(polepoly,[dx,0]);
    polepoly = rotate(polepoly,rad2deg(dtheta),[x,0.25/2]);
    % Update the cart and pole positions on the plot
    cartplot.Shape = cartpoly;
    poleplot.Shape = polepoly;
    % Refresh rendering in the figure window
    drawnow();
    pause(0.02)
end


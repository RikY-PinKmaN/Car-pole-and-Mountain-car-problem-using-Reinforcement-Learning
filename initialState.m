function [State] = initialState()
% initialise the state [X, Xdot, Theta, Thetadot]
State = [0; 0; 2*0.05*rand-0.05; 0]; % Theta (+- .05 rad), else zero
function [next_j] = stateIdx(State,Aux)
% Function to extract the state index for a given state value:
%
% function [next_j] = stateIdx(State)
% State is the system state
% Aux contains auxiliary variables
% unpack auxiliary variables
nquant = Aux.nquant;
delta = Aux.delta;
fmin = Aux.fmin;
% discretise the continuous state using quantization (ignore x)
idx = floor((State(2:4) - fmin)./delta)+1;
next_j = sub2ind([nquant nquant nquant],idx(1),idx(2),idx(3))
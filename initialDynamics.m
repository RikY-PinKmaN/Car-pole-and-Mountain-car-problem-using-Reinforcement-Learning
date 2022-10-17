function [InitialObservation,LoggedSignals] = initialDynamics()
xmin = -0.6;
xmax= -0.4;

x = xmin + (xmax-xmin)*rand;
xdot = 0;

InitialObservation = [x, xdot]';
LoggedSignals = [x, xdot]';


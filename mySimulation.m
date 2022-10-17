function [State,Reward,flag] = mySimulation(State,A)
% Function to implement cart pole dynamics
%
% function [State,Reward,flag] = mySimulation(State,A)
% cartpole parameters
Gravity = 9.8; % Acceleration due to gravity in m/s^2
CartMass = 1.0; % Mass of the cart
PoleMass = 0.1; % Mass of the pole
HalfPoleLength = 0.5; % Half the length of the pole
Ts = 0.02; % Sample time
AngleThreshold = 30 * pi/180; % Angle at which to fail the episode (radians)
DisplacementThreshold = 2.4; % Distance at which to fail the episode
RewardForNotFalling = 1; % Reward each time step the cart-pole is balanced
PenaltyForFalling = -10; % Penalty when the cart-pole fails to balance
flag = 0; % flag for termination
% Get action
Force = A;
% Unpack state vector
X = State(1);
XDot = State(2);
Theta = State(3);
ThetaDot = State(4);
% Cache to avoid recomputation
CosTheta = cos(Theta);
SinTheta = sin(Theta);
SystemMass = CartMass + PoleMass;
temp = (Force + PoleMass*HalfPoleLength*ThetaDot^2*SinTheta)/SystemMass;
% Apply motion equations
ThetaDotDot = (Gravity*SinTheta - CosTheta*temp)...
/ (HalfPoleLength*(4.0/3.0 - PoleMass*CosTheta*CosTheta/SystemMass));
XDotDot = temp - PoleMass*HalfPoleLength*ThetaDotDot*CosTheta/SystemMass;
% Euler integration
State = State + Ts.*[XDot;XDotDot;ThetaDot;ThetaDotDot];
% Check terminal condition
X = State(1);
Theta = State(3);
if abs(X) > DisplacementThreshold || abs(Theta) > AngleThreshold
Reward = PenaltyForFalling;
flag = 1;
else
Reward = RewardForNotFalling;
end
% the end
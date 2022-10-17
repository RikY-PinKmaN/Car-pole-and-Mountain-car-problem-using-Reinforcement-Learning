function [NextObs,Reward,IsDone,LoggedSignals] = stepDynamics(Action,LoggedSignals)
IsDone = false;

x_t = LoggedSignals(1);
xdot_t = LoggedSignals(2);

u_t = Action;

xddot_t = (0.001*u_t) - (0.0025*cos(3*x_t));

T = 1;
x_tp1 = x_t + (T*xdot_t);
xdot_tp1 = xdot_t + (T*xddot_t);

if x_tp1 < -1.2
    x_tp1 = -1.2;
elseif x_tp1 > 0.5
    IsDone = true;
end

if xdot_tp1 < -0.07
    xdot_tp1 = -0.07;
elseif xdot_tp1 > 0.07
    xdot_tp1 = 0.07;
end

NextObs = [x_tp1, xdot_tp1]';

LoggedSignals = [x_tp1, xdot_tp1]';

if IsDone
    Reward = -1;
else
    Reward = -1;
end


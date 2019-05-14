%% Read Step Tracking Log File
clear;clc;

filename = 'log_stepTracking.csv';
log_csv = csvread(filename);

% [ y1,u1, episode_reward ]
res_y1 = log_csv(:,1);
res_u1 = log_csv(:,2);
refs = [0.35,0.30,0.45,0.55,0.30];
% len_ref = length(references);
t = linspace(0,5,500*5);
f = @(t) refs(1)*(t > 0 & t < 1) + refs(2)*( t > 1 & t < 2)+ refs(3)*(t > 2 & t < 3) + ...
refs(4)*(t > 3 & t < 4) + refs(5)*(t > 4 & t < 5);

figure(1);
plot(res_y1,'r');
hold on
plot(f(t),'g-');
title('Outputs in Step Tracking')
legend('Output-1','Reference');
hold off

% 
% figure(2);
% plot(res_u1,'r');
% title('Control Signals in Step Tracking')
% legend('Input-1');
% hold off
%% Read CSV File and PLOT
clear;
clc;

%% Read y1,y2,u1,u2

%  Find Best Episode..
filename1 = 'log_EpisodeReward.csv';
log_Reward_csv = csvread(filename1);
episode_rewards = log_Reward_csv(:,1);
[reward_max,best_episode_number]= max(episode_rewards);

% [ y1, y2, u1, u2, episode_reward ]
filename2 = 'log.csv';
log_csv = csvread(filename2);

% row_min = 300*(best_episode_number-1)+1 ;
max_steps = 200;
row_min = max_steps*(best_episode_number-1)+1 ;
row_max = row_min + max_steps - 1 ;
best_episode = (log_csv(row_min:row_max,:));


y1 = best_episode(:,1);
u1 = best_episode(:,2);
r = best_episode(:,3);



figure(1);
plot(y1);
title("( Best Episode : " + int2str(best_episode_number) + " ) Output ")

figure(2);
plot(u1);
title("( Best Episode : "  + int2str(best_episode_number) + " ) Control Signal ")

figure(5);
episode_rewards_normalized = (episode_rewards - min(episode_rewards)) / ( max(episode_rewards) - min(episode_rewards) );
plot(episode_rewards_normalized);
title('Normalized Episode Rewards')



%% Read Step Respnonse

filename4 = 'log_output.csv';
log_csv2 = csvread(filename4);

% [ y1, y2, u1, u2, episode_reward ]
res_y1 = log_csv2(:,1);
res_u1 = log_csv2(:,2);

figure(6);
plot(res_y1);
title('Step Response of Output')


figure(7);
plot(res_u1);
title('Control Signal ( Step Response )')

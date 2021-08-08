function [modelParameters] = positionEstimatorTraining(training_data) 
  % - training_data:
  %     training_data(n,k)              (n = training_data id,  k = reaching angle)
  %     training_data(n,k).training_dataId      unique number of the training_data
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model

%% Feature extraction for reaching angle classifier
ra_features = [];
ra_labels = (1:8)';

for k = 1: length(training_data(1,:)) % loop for each reaching angle
    spike_rate = [];
    for  n = 1 : length(training_data(:,1)) % loop for each training_data
        sr_dt = [];
        for a = 1 :4
            sr = mean(training_data(n,k).spikes(:,a*80-79:a*80),2);
            sr_dt = [sr_dt, sr'];
        end       
        spike_rate = cat(1, spike_rate, sr_dt);
    end
    ra_features = cat(1, ra_features, mean(spike_rate));
end

% Training a model for reaching angle classifier
modelParameters.cecoc = fitcecoc(ra_features, ra_labels); 

%% Feature extraction for x,y velocity prediction
dt = 20; 
spike_rates = [];
sr_dt = [];
xvel = [];
yvel = [];
xvel_features = [];
yvel_features = [];

for k = 1: length(training_data(1,:)) % loop for each reaching angle
    for  n = 1 : length(training_data(:,1)) % loop for each training_data
        for t = 320:10:540
                x_min = training_data(n,k).handPos(1,t);
                x_max = training_data(n,k).handPos(1,t+dt);
                x_vel = (x_max - x_min)/(dt*0.001);
                
                y_min = training_data(n,k).handPos(2,t);
                y_max = training_data(n,k).handPos(2,t+dt);
                y_vel = (y_max - y_min)/(dt*0.001);
                
                xvel = cat(1,xvel, x_vel);
                yvel = cat(1,yvel, y_vel);

            for i = 1 : 98  % loop for each neuron                
                %spikes rates, 20ms intervals for 320ms from 540
                sr_dt = cat(2, sr_dt, mean(training_data(n,k).spikes(i,t-10:t+10))/0.001);                      
            end
            spike_rates = cat(1, spike_rates, sr_dt); 
            sr_dt = [];
        end
        xvel_features = cat(1, xvel_features, xvel);
        xvel = [];
        yvel_features = cat(1, yvel_features, yvel);
        yvel = [];
    end
    
 % Training x,y velocity regressor for each reaching angle   
 if k == 1
     modelParameters.regr1_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr1_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 2
     modelParameters.regr2_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr2_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 3
     modelParameters.regr3_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr3_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 4
     modelParameters.regr4_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr4_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 5
     modelParameters.regr5_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr5_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 6
     modelParameters.regr6_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr6_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 7
     modelParameters.regr7_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr7_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 elseif k == 8
     modelParameters.regr8_x = fitlm(spike_rates, xvel_features);
     modelParameters.regr8_y = fitlm(spike_rates, yvel_features);
     spike_rates = [];
     xvel_features = [];
     yvel_features = [];
 end
end

% Calculate average trajectory upto 560 ms
xtraj(8).handPos = [];
ytraj(8).handPos = [];

for k = 1 : length(training_data(1, :))% each angle
    x = [];
    y = [];
    
    for n = 1: length(training_data(:,1)) % each trial
        x = cat(1, x, training_data(n,k).handPos(1,1:560));
        y = cat(1, y, training_data(n,k).handPos(2,1:560));   
    end
    xtraj(k).handPos = x;
    ytraj(k).handPos = y;
end

% Caculate x,y min/max at each time stamp for each angle
for k = 1 : length(training_data(1, :))% each angle
    t = 320:20:540;
    
    modelParameters.xtr_max(k).handPos = max(xtraj(k).handPos(:,t));
    modelParameters.ytr_max(k).handPos = max(ytraj(k).handPos(:,t));
    
    modelParameters.xtr_min(k).handPos = min(xtraj(k).handPos(:,t));
    modelParameters.ytr_min(k).handPos = min(ytraj(k).handPos(:,t));
end

  % mean trajectories for each reaching angle
  % Return Value:
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
 
end

function [x, y,newModelParameters] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 300):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x300 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
%% STORE MODEL PARAMETERS FOR OUTPUT
newModelParameters=modelParameters;

%% VALUE STORE
lgth=length(test_data.spikes);

%% ANGLE PREDICTION
if lgth<=320   
    ra_test = [];
    for a = 1 :4
        ra_test = [ra_test, mean(test_data.spikes(:, a*80-79: a*80),2)'];
    end  
    newModelParameters.ra_label = predict(modelParameters.cecoc, ra_test);
end
%% X Y VELOCITY
spike_rates_test = [];

if lgth > 520
    tmax=520;
else
    tmax=lgth;
end 

for i = 1 : 98  % loop for each neuron
        spike_rates_test = cat(2, spike_rates_test, mean(test_data.spikes(i,tmax-40:tmax-20))/0.001);
end

if newModelParameters.ra_label == 1
    x_pred = predict(modelParameters.regr1_x, spike_rates_test);
    y_pred = predict(modelParameters.regr1_y, spike_rates_test);
    
elseif newModelParameters.ra_label == 2
    x_pred = predict(modelParameters.regr2_x, spike_rates_test);
    y_pred = predict(modelParameters.regr2_y, spike_rates_test);
    
elseif newModelParameters.ra_label == 3
    x_pred = predict(modelParameters.regr3_x, spike_rates_test);
    y_pred = predict(modelParameters.regr3_y, spike_rates_test);
    
elseif newModelParameters.ra_label == 4
    x_pred = predict(modelParameters.regr4_x, spike_rates_test);
    y_pred = predict(modelParameters.regr4_y, spike_rates_test);
     
elseif newModelParameters.ra_label == 5
    x_pred = predict(modelParameters.regr5_x, spike_rates_test);
    y_pred = predict(modelParameters.regr5_y, spike_rates_test);
     
elseif newModelParameters.ra_label == 6
    x_pred = predict(modelParameters.regr6_x, spike_rates_test);
    y_pred = predict(modelParameters.regr6_y, spike_rates_test);
    
elseif newModelParameters.ra_label == 7
    x_pred = predict(modelParameters.regr7_x, spike_rates_test);
    y_pred = predict(modelParameters.regr7_y, spike_rates_test);
     
elseif newModelParameters.ra_label == 8
    x_pred = predict(modelParameters.regr8_x, spike_rates_test);
    y_pred = predict(modelParameters.regr8_y, spike_rates_test);
end

%% X, Y POSITION

if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
%     if x > modelParameters.xtr_max(newModelParameters.ra_label).handPos(1)
%         x = modelParameters.xtr_max(newModelParameters.ra_label).handPos(1);
%     elseif x < modelParameters.xtr_min(newModelParameters.ra_label).handPos(1)
%         x = modelParameters.xtr_min(newModelParameters.ra_label).handPos(1);
%     end      
    y = test_data.startHandPos(2);
%     if y > modelParameters.ytr_max(newModelParameters.ra_label).handPos(1)
%         y = modelParameters.ytr_max(newModelParameters.ra_label).handPos(1);
%     elseif y < modelParameters.ytr_min(newModelParameters.ra_label).handPos(1)
%         y = modelParameters.ytr_min(newModelParameters.ra_label).handPos(1);
%     end
     
elseif 320 < length(test_data.spikes) && length(test_data.spikes) <= 520
    % position = previous position + velocity * 20ms
    x=test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + x_pred*(20*0.001);
    if x > modelParameters.xtr_max(newModelParameters.ra_label).handPos((tmax-300)/20)
        x = modelParameters.xtr_max(newModelParameters.ra_label).handPos((tmax-300)/20);
    elseif x < modelParameters.xtr_min(newModelParameters.ra_label).handPos((tmax-300)/20)
        x = modelParameters.xtr_min(newModelParameters.ra_label).handPos((tmax-300)/20);
    end
    
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + y_pred*(20*0.001);
    if y > modelParameters.ytr_max(newModelParameters.ra_label).handPos((tmax-300)/20)
        y = modelParameters.ytr_max(newModelParameters.ra_label).handPos((tmax-300)/20);
    elseif y < modelParameters.ytr_min(newModelParameters.ra_label).handPos((tmax-300)/20)
        y = modelParameters.ytr_min(newModelParameters.ra_label).handPos((tmax-300)/20);
    end    
  
elseif length(test_data.spikes) > 520
    x=test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:)));
    if x > modelParameters.xtr_max(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos))
        x = modelParameters.xtr_max(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos));
    elseif x < modelParameters.xtr_min(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos))
        x = modelParameters.xtr_min(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos));
    end    
    
    y=test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:)));
    if y > modelParameters.ytr_max(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos))
        y = modelParameters.ytr_max(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos));
    elseif y < modelParameters.ytr_min(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos))
        y = modelParameters.ytr_min(newModelParameters.ra_label).handPos(length(modelParameters.xtr_max(1).handPos));
    end    
end

  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end

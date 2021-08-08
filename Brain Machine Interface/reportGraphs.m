load monkeydata_training.mat;
trial_n = 1;
angle = 7;
neurons = [34, 90];

font_size = 15;
f = figure;
f.Position = [0, 0, 675, 850];
%%
tiledlayout(4,1);
ax1 = nexttile;
ax2 = nexttile;
ax3 = nexttile;
ax4 = nexttile;


%% single-trial raster plot

raster = [];
for t = 1:length(trial(trial_n, angle).spikes(1,:))
    q = trial(trial_n, angle).spikes(:,t);
    SP = find(q);                                      % find the spike times
    raster=[raster;t*ones(length(SP),1),SP];           % save spike times for plotting
end

plot(ax1, raster(:,1), raster(:,2),'.b')
xlabel(ax1, 'Time (ms)','fontsize',font_size)
ylabel(ax1, 'Neuron','fontsize',font_size)
title(ax1, 'A')

set(ax1,'fontsize',font_size);
%set(ax1,'YDir','normal')

%% single neuron rate-time
hold(ax2, 'on')
legend(ax2)
legend(ax2, 'boxoff')
for neuron = neurons
    sigma = 20;
    a = [];
    for n = 1 : length(trial(:,angle))
        a = [a, find(trial(n,angle).spikes(neuron,:))];
    end
    sig = zeros(length(trial(trial_n, angle).spikes(neuron,:)),1);
    for t = 1:length(sig)
        sig(t) = 10*sum( ((1/(((2*pi)^(1/2))*sigma)) * exp(- ((t-a).^2)/(2*(sigma^2)) ) ) ); % factor of 10 comes from 1000 ms / 100 trials
    end
    text = ['Neuron ' , int2str(neuron)];
    plot(ax2, sig, 'DisplayName', text, 'LineWidth', 2)

end
xlabel(ax2, 'Time (ms)','fontsize',font_size)
ylabel(ax2, 'Rate (Hz)','fontsize',font_size)
title(ax2, 'B')
set(ax2,'fontsize',font_size);

%% Arm Position
handPos = trial(trial_n,angle).handPos;

hold(ax3, 'on')
plot(ax3, handPos(1,:), 'LineWidth', 2)
plot(ax3, handPos(2,:), 'LineWidth', 2)
legend(ax3, 'x position', 'y position')
legend(ax3, 'boxoff')

ylabel(ax3, 'Displacement','fontsize',font_size)
xlabel(ax3, 'Time (ms)','fontsize',font_size)
title(ax3, 'C')
set(ax3,'fontsize',font_size);


%% Tuning curve
hold(ax4, 'on')

for neuron = neurons
    tuning_curve = [];
    errors = [];
    for angle = 1:8
        sr_list = [];
        for n = 1:length(trial(:,1))
            sr_list = [sr_list, mean(trial(n,angle).spikes(neuron,:))*1000];
        end

        tuning_curve = [tuning_curve, mean(sr_list)];
        errors = [errors, std(sr_list)];
    end
    text = ['Neuron ' , int2str(neuron)];
    errorbar(ax4, tuning_curve, errors, 'DisplayName', text, 'LineWidth', 2)

end
legend(ax4, 'Location', 'West')
legend('boxoff')
ylabel(ax4, 'Rate (Hz)','fontsize',font_size)
xlabel(ax4, 'Reaching angle','fontsize',font_size)
title(ax4, 'D')
set(ax4,'fontsize',font_size);




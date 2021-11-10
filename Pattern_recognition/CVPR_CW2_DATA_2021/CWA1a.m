%% Load data from acrylic for pressure, vibration and temperature into arrays
clc;
clear all;

% Timestep for electrodes. This was implemented last in the code and is
% here only due to convinience. Please ignore in case if seems confusing
time_step = 400;

% Arrays for the two fingers for accrylic

acc0_P = zeros(10,1000);
acc0_V = zeros(10,1000);
acc0_T = zeros(10,1000);
acc1_P = zeros(10,1000);
acc1_V = zeros(10,1000);
acc1_T = zeros(10,1000);
acc1_elec = zeros(19,10);

%Collect the data for Acrylic
for i=1:9
    name = 'PR_CW_DATA_2021\acrylic_211_0'+ string(i)+'_HOLD.mat';
    load(name);
    acc0_P(i,:) = F0pdc;
    acc1_P(i,:) = F1pdc;
    acc0_V(i,:) = F0pac(2,:);
    acc1_V(i,:) = F1pac(2,:);
    acc0_T(i,:) = F0tdc;
    acc1_T(i,:) = F1tdc;
    acc1_elec(:, i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\acrylic_211_10_HOLD.mat')
acc0_P(10,:) = F0pdc;
acc1_P(10,:) = F1pdc;
acc0_V(10,:) = F0pac(2,:);
acc1_V(10,:) = F1pac(2,:);
acc0_T(10,:) = F0tdc;
acc1_T(10,:) = F1tdc;
acc1_elec(:, 10) = F1Electrodes(:, time_step);


% Black foam arrays
black0_P = zeros(10,1000);
black0_V = zeros(10,1000);
black0_T = zeros(10,1000);
black1_P = zeros(10,1000);
black1_V = zeros(10,1000);
black1_T = zeros(10,1000);
black1_elec = zeros(19, 10);

% Collection of data for Black Foam
for i=1:9
    name = 'PR_CW_DATA_2021\black_foam_110_0'+ string(i)+'_HOLD.mat';
    load(name);
    black0_P(i,:) = F0pdc(1:1000);
    black1_P(i,:) = F1pdc(1:1000);
    black0_V(i,:) = F0pac(2,1:1000);
    black1_V(i,:) = F1pac(2,1:1000);
    black0_T(i,:) = F0tdc(1:1000);
    black1_T(i,:) = F1tdc(1:1000);
    black1_elec(:, i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\black_foam_110_10_HOLD.mat')
black0_P(10,:) = F0pdc(1:1000);
black1_P(10,:) = F1pdc(1:1000);
black0_V(10,:) = F0pac(2,1:1000);
black1_V(10,:) = F1pac(2,1:1000);
black0_T(10,:) = F0tdc(1:1000);
black1_T(10,:) = F1tdc(1:1000);
black1_elec(:,10) = F1Electrodes(:, time_step);


% Car sponge arrays
car0_P = zeros(10,1000);
car0_V = zeros(10,1000);
car0_T = zeros(10,1000);
car1_P = zeros(10,1000);
car1_V = zeros(10,1000);
car1_T = zeros(10,1000);
car1_elec = zeros(19, 10);

%Collect the data for Car Sponge
for i=1:9
    name = 'PR_CW_DATA_2021\car_sponge_101_0'+ string(i)+'_HOLD.mat';
    load(name);
    car0_P(i,:) = F0pdc;
    car1_P(i,:) = F1pdc;
    car0_V(i,:) = F0pac(2,:);
    car1_V(i,:) = F1pac(2,:);
    car0_T(i,:) = F0tdc;
    car1_T(i,:) = F1tdc;
    car1_elec(:,i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\car_sponge_101_10_HOLD.mat')
car0_P(10,:) = F0pdc;
car1_P(10,:) = F1pdc;
car0_V(10,:) = F0pac(2,:);
car1_V(10,:) = F1pac(2,:);
car0_T(10,:) = F0tdc;
car1_T(10,:) = F1tdc;
car1_elec(:, 10) = F1Electrodes(:, time_step);

% Flour sack arrays
flour0_P = zeros(10,1000);
flour0_V = zeros(10,1000);
flour0_T = zeros(10,1000);
flour1_P = zeros(10,1000);
flour1_V = zeros(10,1000);
flour1_T = zeros(10,1000);
flour1_elec = zeros(19,10);

%Collect the data for Flour sack
for i=1:9
    name = 'PR_CW_DATA_2021\flour_sack_410_0'+ string(i)+'_HOLD.mat';
    load(name);
    flour0_P(i,:) = F0pdc;
    flour1_P(i,:) = F1pdc;
    flour0_V(i,:) = F0pac(2,:);
    flour1_V(i,:) = F1pac(2,:);
    flour0_T(i,:) = F0tdc;
    flour1_T(i,:) = F1tdc;
    flour1_elec(:,i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\flour_sack_410_10_HOLD.mat')
flour0_P(10,:) = F0pdc;
flour1_P(10,:) = F1pdc;
flour0_V(10,:) = F0pac(2,:);
flour1_V(10,:) = F1pac(2,:);
flour0_T(10,:) = F0tdc;
flour1_T(10,:) = F1tdc;
flour1_elec(:, 10) = F1Electrodes(:, time_step);


% Kitchen Sponge Data
kitchen0_P = zeros(10,1000);
kitchen0_V = zeros(10,1000);
kitchen0_T = zeros(10,1000);
kitchen1_P = zeros(10,1000);
kitchen1_V = zeros(10,1000);
kitchen1_T = zeros(10,1000);
kitchen1_elec = zeros(19, 10);

%Collect the data for Flour sack
for i=1:9
    name = 'PR_CW_DATA_2021\kitchen_sponge_114_0'+ string(i)+'_HOLD.mat';
    load(name);
    kitchen0_P(i,:) = F0pdc;
    kitchen1_P(i,:) = F1pdc;
    kitchen0_V(i,:) = F0pac(2,:);
    kitchen1_V(i,:) = F1pac(2,:);
    kitchen0_T(i,:) = F0tdc;
    kitchen1_T(i,:) = F1tdc;
    kitchen1_elec(:, i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\kitchen_sponge_114_10_HOLD.mat')
kitchen0_P(10,:) = F0pdc;
kitchen1_P(10,:) = F1pdc;
kitchen0_V(10,:) = F0pac(2,:);
kitchen1_V(10,:) = F1pac(2,:);
kitchen0_T(10,:) = F0tdc;
kitchen1_T(10,:) = F1tdc;
kitchen1_elec(:, 10) = F1Electrodes(:, time_step);





%
steel0_P = zeros(10,1000);
steel0_V = zeros(10,1000);
steel0_T = zeros(10,1000);
steel1_P = zeros(10,1000);
steel1_V = zeros(10,1000);
steel1_T = zeros(10,1000);
steel1_elec = zeros(19, 10);

%Collect the data for Acrylic
for i=1:9
    name = 'PR_CW_DATA_2021\steel_vase_702_0'+ string(i)+'_HOLD.mat';
    load(name);
    steel0_P(i,:) = F0pdc;
    steel1_P(i,:) = F1pdc;
    steel0_V(i,:) = F0pac(2,:);
    steel1_V(i,:) = F1pac(2,:);
    steel0_T(i,:) = F0tdc;
    steel1_T(i,:) = F1tdc;
    steel1_elec(:, i) = F1Electrodes(:, time_step);
end
load('PR_CW_DATA_2021\steel_vase_702_10_HOLD.mat')
steel0_P(10,:) = F0pdc;
steel1_P(10,:) = F1pdc;
steel0_V(10,:) = F0pac(2,:);
steel1_V(10,:) = F1pac(2,:);
steel0_T(10,:) = F0tdc;
steel1_T(10,:) = F1tdc;
steel1_elec(:, 10) = F1Electrodes(:, time_step);


%%
% Data edit to be used in plots
Acc0P = acc0_P';
Acc0V = acc0_V';
Acc0T = acc0_T';
Acc1P = acc1_P';
Acc1V = acc1_V';
Acc1T = acc1_T';
Black0P = black0_P';
Black0V = black0_V';
Black0T = black0_T';
Black1P = black1_P';
Black1V = black1_V';
Black1T = black1_T';
Car0P = car0_P';
Car0V = car0_V';
Car0T = car0_T';
Car1P = car1_P';
Car1V = car1_V';
Car1T = car1_T';
Flour0P = flour0_P';
Flour0V = flour0_V';
Flour0T = flour0_T';
Flour1P = flour1_P';
Flour1V = flour1_V';
Flour1T = flour1_T';
Kitchen0P = kitchen0_P';
Kitchen0V = kitchen0_V';
Kitchen0T = kitchen0_T';
Kitchen1P = kitchen1_P';
Kitchen1V = kitchen1_V';
Kitchen1T = kitchen1_T';
Steel0P = steel0_P';
Steel0V = steel0_V';
Steel0T = steel0_T';
Steel1P = steel1_P';
Steel1V = steel1_V';
Steel1T = steel1_T';


%End of data collection

%% Plot the data to visualize how the PVT changes for a single cycle

% Squeeze cycle iteration. Change this number to see other data collected
n = 1;
% Default is set to one, for the first time the sensors touch the material
% and sceeze. Further iterations are for after the sensors already touch
% the material, but squeeze further.

figure('Name', 'Data from sensor 0')
subplot(3,1,1)
hold on;
title('Pressure reading from F0')
plot(Acc0P(:,n))
plot(Black0P(:,n))
plot(Car0P(:,n))
plot(Flour0P(:,n))
plot(Kitchen0P(:,n))
plot(Steel0P(:,n))

subplot(3,1,2)
hold on;
title('Vibration reading from F0')
plot(Acc0V(:,n))
plot(Black0V(:,n))
plot(Car0V(:,n))
plot(Flour0V(:,n))
plot(Kitchen0V(:,n))
plot(Steel0V(:,n))

subplot(3,1,3)
hold on;
title('Temperature reading from F0')
plot(Acc0T(:,n))
plot(Black0T(:,n))
plot(Car0T(:,n))
plot(Flour0T(:,n))
plot(Kitchen0T(:,n))
plot(Steel0T(:,n))

figure('Name', 'Data from sensor 1')
subplot(3,1,1)
hold on;
title('Pressure reading from F1')
plot(Acc1P(:,n))
plot(Black1P(:,n))
plot(Car1P(:,n))
plot(Flour1P(:,n))
plot(Kitchen1P(:,n))
plot(Steel1P(:,n))

subplot(3,1,2)
hold on;
title('Vibration reading from F1')
plot(Acc1V(:,n))
plot(Black1V(:,n))
plot(Car1V(:,n))
plot(Flour1V(:,n))
plot(Kitchen1V(:,n))
plot(Steel1V(:,n))

subplot(3,1,3)
hold on;
title('Temperature reading from F1')
plot(Acc1T(:,n))
plot(Black1T(:,n))
plot(Car1T(:,n))
plot(Flour1T(:,n))
plot(Kitchen1T(:,n))
plot(Steel1T(:,n))

%%
hold off

figure(5)
plot(F1Electrodes')
title('F1 electrodes')

figure(6)
plot(F0Electrodes')
title('F0 electrodes')


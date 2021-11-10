%% Specific Time step data
clear all; close all; clc;

CWA1a;

time_step = 400;
%time_step = 16;
% Data edit to be used in plots
Acc0P = acc0_P(:, time_step);
Acc0V = acc0_V(:, time_step);
Acc0T = acc0_T(:, time_step);
Acc1P = acc1_P(:, time_step);
Acc1V = acc1_V(:, time_step);
Acc1T = acc1_T(:, time_step);
Black0P = black0_P(:, time_step);
Black0V = black0_V(:, time_step);
Black0T = black0_T(:, time_step);
Black1P = black1_P(:, time_step);
Black1V = black1_V(:, time_step);
Black1T = black1_T(:, time_step);
Car0P = car0_P(:, time_step);
Car0V = car0_V(:, time_step);
Car0T = car0_T(:, time_step);
Car1P = car1_P(:, time_step);
Car1V = car1_V(:, time_step);
Car1T = car1_T(:, time_step);
Flour0P = flour0_P(:, time_step);
Flour0V = flour0_V(:, time_step);
Flour0T = flour0_T(:, time_step);
Flour1P = flour1_P(:, time_step);
Flour1V = flour1_V(:, time_step);
Flour1T = flour1_T(:, time_step);
Kitchen0P = kitchen0_P(:, time_step);
Kitchen0V = kitchen0_V(:, time_step);
Kitchen0T = kitchen0_T(:, time_step);
Kitchen1P = kitchen1_P(:, time_step);
Kitchen1V = kitchen1_V(:, time_step);
Kitchen1T = kitchen1_T(:, time_step);
Steel0P = steel0_P(:, time_step);
Steel0V = steel0_V(:, time_step);
Steel0T = steel0_T(:, time_step);
Steel1P = steel1_P(:, time_step);
Steel1V = steel1_V(:, time_step);
Steel1T = steel1_T(:, time_step);


%End of data collection

%% At specific Time step

% Squeeze cycle iteration. Change this number to see other data collected

% Default is set to one, for the first time the sensors touch the material
% and sceeze. Further iterations are for after the sensors already touch
% the material, but squeeze further.

figure('Name', 'Data from sensor 0')
subplot(3,1,1)
hold on;
plot(Acc0P)
plot(Black0P)
plot(Car0P)
plot(Flour0P)
plot(Kitchen0P)
plot(Steel0P)

subplot(3,1,2)
hold on;
plot(Acc0V)
plot(Black0V)
plot(Car0V)
plot(Flour0V)
plot(Kitchen0V)
plot(Steel0V)

subplot(3,1,3)
hold on;
plot(Acc0T)
plot(Black0T)
plot(Car0T)
plot(Flour0T)
plot(Kitchen0T)
plot(Steel0T)

figure('Name', 'Data from sensor 1')
subplot(3,1,1)
hold on;
plot(Acc1P)
plot(Black1P)
plot(Car1P)
plot(Flour1P)
plot(Kitchen1P)
plot(Steel1P)

subplot(3,1,2)
hold on;
plot(Acc1V)
plot(Black1V)
plot(Car1V)
plot(Flour1V)
plot(Kitchen1V)
plot(Steel1V)

subplot(3,1,3)
hold on;
plot(Acc1T)
plot(Black1T)
plot(Car1T)
plot(Flour1T)
plot(Kitchen1T)
plot(Steel1T)


%%
% Creating Structure Data
F1_PVT.Acc1P = Acc1P;
F1_PVT.Acc1V = Acc1V;
F1_PVT.Acc1T = Acc1T;
F1_PVT.Black1P = Black1P;
F1_PVT.Black1V = Black1V;
F1_PVT.Black1T = Black1T;
F1_PVT.Car1P = Car1P;
F1_PVT.Car1V = Car1V;
F1_PVT.Car1T = Car1T;
F1_PVT.Flour1P = Flour1P;
F1_PVT.Flour1V = Flour1V;
F1_PVT.Flour1T = Flour1T;
F1_PVT.Kitchen1P = Kitchen1P;
F1_PVT.Kitchen1V = Kitchen1V;
F1_PVT.Kitchen1T = Kitchen1T;
F1_PVT.Steel1P = Steel1P;
F1_PVT.Steel1V = Steel1V;
F1_PVT.Steel1T = Steel1T;

save('F1_PVT.mat', 'F1_PVT')


F1_Electrodes.Acc1Elec = acc1_elec;
F1_Electrodes.Black1Elec = black1_elec;
F1_Electrodes.Car1Elec = car1_elec;
F1_Electrodes.Flour1Elec = flour1_elec;
F1_Electrodes.Kitchen1Elec = kitchen1_elec;
F1_Electrodes.Steel1Elec = steel1_elec;

save('F1_Elec.mat', 'F1_Electrodes');




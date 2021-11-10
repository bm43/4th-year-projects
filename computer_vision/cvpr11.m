path='C:\Users\SamSung\Desktop\1.jpg'

%detect calibration pattern
[imagePoints, boardSize]=detectCheckerboardPoints(path);

% generate world coord of corners of sq
squareSize=29;
worldPoints=generateCheckerboardPoints(boardSize, sqaureSize);

%calibrate the camera
[params, ~, estimationErrors] = estimateCameraParameters(iamgePoints, worldPoints);
%%
disp(cameraParams.Intrinsics)
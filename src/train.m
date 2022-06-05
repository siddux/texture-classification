clear;clc;

features0 = [];
features45 = [];
features90 = [];
features135 = [];
features180 = [];
features225 = [];
features270 = [];
features315 = [];
labels = {};
X = [];

rotation = "00";
fileinfo = dir(strcat('../dataset/',rotation,'/'));
fnames = {fileinfo.name}';
fnames(3) = [];
fnames(2) = [];
fnames(1) = [];

for i = 1:length(fnames)
  filename = string(fnames(i));
  disp(filename)

  try
      imageData = rgb2gray(imread(strcat(strcat('../dataset/',rotation,'/'),string(filename))));
      gcm = graycomatrix(imageData,'Offset',[0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1; 1 0; 1 1]);
      features0 = haralickTextureFeatures(gcm(:,:,1), [1,2,5,9])';
      features0(features0 == 0) = [];
      features45 = haralickTextureFeatures(gcm(:,:,2), [1,2,5,9])';
      features45(features45 == 0) = [];
      features90 = haralickTextureFeatures(gcm(:,:,3), [1,2,5,9])';
      features90(features90 == 0) = [];
      features135 = haralickTextureFeatures(gcm(:,:,4), [1,2,5,9])';
      features135(features135 == 0) = [];
      features180 = haralickTextureFeatures(gcm(:,:,5), [1,2,5,9])';
      features180(features180 == 0) = [];
      features225 = haralickTextureFeatures(gcm(:,:,6), [1,2,5,9])';
      features225(features225 == 0) = [];
      features270 = haralickTextureFeatures(gcm(:,:,7), [1,2,5,9])';
      features270(features270 == 0) = [];
      features315 = haralickTextureFeatures(gcm(:,:,8), [1,2,5,9])';
      features315(features315 == 0) = [];
    
      energy = [features0(1),features45(1),features90(1),features135(1),...
          features180(1),features225(1),features270(1),features315(1)];
      contrast = [features0(2),features45(2),features90(2),features135(2),...
          features180(2),features225(2),features270(2),features315(2)];
      homogeneity = [features0(3),features45(3),features90(3),features135(3),...
          features180(3),features225(3),features270(3),features315(3)];
      entropy = [features0(4),features45(4),features90(4),features135(4),...
          features180(4),features225(4),features270(4),features315(4)];
    
      X(end+1,1) = (1/8) * sum(energy);
      X(end,2) = (1/8) * sum(contrast);
      X(end,3) = (1/8) * sum(homogeneity);
      X(end,4) = (1/8) * sum(entropy);
    
      X(end,5) = max(energy) - min(energy);
      X(end,6) = max(contrast) - min(contrast);
      X(end,7) = max(homogeneity) - min(homogeneity);
      X(end,8) = max(entropy) - min(entropy);
    
      X(end,9) = (1/8)*sum(abs(energy - X(end,1)));
      X(end,10) = (1/8)*sum(abs(contrast - X(end,2)));
      X(end,11) = (1/8)*sum(abs(homogeneity - X(end,3)));
      X(end,12) = (1/8)*sum(abs(entropy - X(end,4)));
    
      X(end,13:20) = abs(fft(energy));
      X(end,21:28) = angle(fft(energy));
      X(end,29:36) = abs(fft(contrast));
      X(end,37:44) = angle(fft(contrast));
      X(end,45:52) = abs(fft(homogeneity));
      X(end,53:60) = angle(fft(homogeneity));
      X(end,61:68) = abs(fft(entropy));
      X(end,69:76) = angle(fft(entropy));
    
      label = split(string(filename),"_");
      labels{end+1} = label(1);
  catch
  end
end

labels = cellfun(@char,labels','un',0);

%% Train model

t = templateSVM('Standardize',true,'KernelFunction','gaussian','Solver','L1QP');
MdlSVM = fitcecoc(X,labels,'Learners',t);
save('../models/MdlSVM','MdlSVM');
Mdl1NN = fitcknn(X,labels,'NumNeighbors',1,'Standardize',1);
save('../models/Mdl1NN','Mdl1NN');
Mdl3NN = fitcknn(X,labels,'NumNeighbors',3,'Standardize',1);
save('../models/Mdl3NN','Mdl3NN');
Mdl5NN = fitcknn(X,labels,'NumNeighbors',5,'Standardize',1);
save('../models/Mdl5NN','Mdl5NN');
try
    MdlNB = fitcnb(X,labels);
    save('../models/MdlNB','MdlNB');
catch error
    disp(error)
end


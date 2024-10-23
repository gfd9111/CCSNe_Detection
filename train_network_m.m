% Create the imagedatastore for network training imds for signal and noise

% clear % clear old variables
s112_LS220_snr01 = 29.21; % SNR for s11.2--LS220 at 0.1kpc, follow 1/dist decay
s15_LS220_snr01 = 142.97; % SNR for s15.0--LS220 at 0.1kpc, follow 1/dist decay
s20_LS220_snr01 = 58.79; % SNR for s20.0--LS220 at 0.1kpc, follow 1/dist decay
s25_LS220_snr01 = 230.75; % SNR for s25.0--LS220 at 0.1kpc, follow 1/dist decay
s40_LS220_snr01 = 83.2; % SNR for s40.0--LS220 at 0.1kpc, follow 1/dist decay
s15_GShen_snr01 = 24.64; % SNR for s15.0--GShen at 0.1kpc, follow 1/dist decay
s15_SFHo_snr01 = 39.62; % SNR for s15.0--SFHo at 0.1kpc, follow 1/dist decay
s20_SFHo_snr01 = 21.71; % SNR for s20.0--SFHo at 0.1kpc, follow 1/dist decay

img_size = [224 224 3];
time_vect = linspace(0, 2, img_size(2)); % time vector
freq_vect = linspace(2^13, 0, img_size(1)); % frequency vector - reverse because of image display
test_path = fullfile("new_simulation_testing_spgram/");
imds_test = imageDatastore(test_path, "IncludeSubfolders", true, "FileExtensions", ".png", "LabelSource", "foldernames");
imds_test = shuffle(imds_test);
categories(imds_test.Labels)
countlabels(imds_test.Labels)
augimds_test = augmentedImageDatastore(img_size, imds_test, "ColorPreprocessing", "gray2rgb"); % use for testing
% X and Y translation for time and frequency uncertainty
aug = imageDataAugmenter("RandXTranslation", 0.5*[-1 1]*img_size(2)); %, "RandYTranslation", [-1 1]*0.05*img_size(1));
validation_path = fullfile("new_simulation_validation_spgram/");
imds_validation = imageDatastore(validation_path, "IncludeSubfolders", true, "FileExtensions", ".png", "LabelSource", "foldernames");
imds_validation = shuffle(imds_validation);
categories(imds_validation.Labels)
countlabels(imds_validation.Labels)
augimds_validation = augmentedImageDatastore(img_size, imds_validation, "ColorPreprocessing", "gray2rgb", "DataAugmentation", aug); % use for validation

% for the training set
imds_train = imageDatastore("new_simulation_training_spgram/", "IncludeSubfolders", true, "FileExtensions", ".png", "LabelSource", "foldernames");
imds_train = shuffle(imds_train);
categories(imds_train.Labels)
countlabels(imds_train.Labels)
N = length(imds_train.Files);
% split imds into train and validation
%[imds_train,imds_validation] = splitEachLabel(imds,0.1,'randomize');
augimds_train = augmentedImageDatastore(img_size, imds_train, "ColorPreprocessing", "gray2rgb", "DataAugmentation", aug); % for training
%augimds_validation = augmentedImageDatastore(img_size, imds_validation, "ColorPreprocessing", "none"); % for param tunning

imds_final = imageDatastore([imds_train.Files; imds_validation.Files], "IncludeSubfolders", true, "FileExtensions", ".png", "LabelSource", "foldernames");
categories(imds_final.Labels)
countlabels(imds_final.Labels)
augimds_final = augmentedImageDatastore(img_size, imds_final, "ColorPreprocessing", "gray2rgb", "DataAugmentation", aug); % use for training after hyper-param tunning
% Proceed to network training here

% Once finish hyper-parameters tunning, use augimds_final for final
% training using all training data

% Using pre-trained network
trainingSetup = load("initial_params_resnet18.mat");
lgraph = initialize_network(trainingSetup);

% Using customized network
% layers = [
%     imageInputLayer([200 200 1],"Name","imageinput","Normalization","zerocenter")
% 
%     convolution2dLayer([3 3],64,"Name","conv1","Padding","same")
%     reluLayer("Name","relu1")
%     averagePooling2dLayer([2 2], 'Stride', [2 2])
% 
%     convolution2dLayer([3 3],32,"Name","conv2","Padding","same")
%     reluLayer("Name","relu2")
%     averagePooling2dLayer([2 2], 'Stride', [2 2])
% 
%     fullyConnectedLayer(2,"Name","fc2")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% lgraph = layerGraph(layers);
% analyzeNetwork(lgraph)

% training options
opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",1e-6,...
    "L2Regularization",0.05,...
    "LearnRateSchedule","none",...
    "LearnRateDropFactor",0.7,...
    "LearnRateDropPeriod",3,...
    "MaxEpochs",10,...
    "MiniBatchSize",128,...
    "OutputNetwork","auto",...
    "Shuffle","every-epoch",...
    "ValidationFrequency",60,...
    "Plots","training-progress",...
    "Verbose", true);

% train network
[net, traininfo] = trainNetwork(augimds_final,lgraph,opts);
% plot training progress
figure, subplot(2,1,1)
plot(traininfo.TrainingAccuracy), ylabel("Accuracy [\%]"), title("Training Graph"), hold on, grid on
% plot(traininfo.ValidationAccuracy, 'Marker', 'o', 'MarkerFaceColor', 'black', 'MarkerEdgeColor','none', 'LineStyle','none'), legend("Train", "Validation")
subplot(2,1,2)
plot(traininfo.TrainingLoss), xlabel("Iteration"), ylabel("Loss"), hold on, grid on
% plot(traininfo.ValidationLoss, 'Marker', 'o', 'MarkerFaceColor', 'black', 'MarkerEdgeColor','none', 'LineStyle','none'), legend("Train", "Validation")
% Test the performance

% for DAG Network from standard training loop
N_test = length(imds_test.Files);
[~, scoresList] = classify(net, augimds_test); % columns: [non-signal signal]
threshold = 0.5;

% summarize distance composition
dist_list = [];
snr_list = [];
s15_TP_list = []; % list of correctly predicted signal for s15.0--SFHo
s20_TP_list = []; % list of correctly predicted signal for s20.0--SFHo
predLabels = strings(N_test,1);
for i = 1:length(augimds_test.Files)
    [~,name,~] = fileparts(augimds_test.Files{i});
    if (imds_test.Labels(i) == "signal")
        dist_str = extractBefore(name, "kpc");
        dist_str = extractAfter(dist_str, "_");
        dist = str2double(dist_str);
        dist_list(end+1) = dist; % append to the distance list, TP + FN
        if scoresList(i,2) >= threshold % all positives
            if contains(name, "s15.0")
                s15_TP_list(end+1) = dist; % TP 15
            elseif contains(name, "s20.0")
                s20_TP_list(end+1) = dist; % TP 20
            end
        end
    end
    
    if scoresList(i,2) >= threshold
        predLabels(i)  = "signal";
    else
        predLabels(i) = "noise";
    end
end
predLabels = categorical(predLabels); % convert to categorical
kpc_dist = unique(dist_list);
figure, histogram(dist_list, kpc_dist), title("distance distribution")
xlabel("Distance [kpc]"), ylabel("Count"), grid on

s15_N_correct_pred_signal_list = zeros(length(kpc_dist),1);
for i = 1:length(kpc_dist)
    kpc_i = kpc_dist(i);
    s15_N_correct_pred_signal_list(i) = numel(find(s15_TP_list == kpc_i));
end
s15_prop_correct_pred_signal = s15_N_correct_pred_signal_list; % percentage of TP for s15

s20_N_correct_pred_signal_list = zeros(length(kpc_dist),1);
for i = 1:length(kpc_dist)
    kpc_i = kpc_dist(i);
    s20_N_correct_pred_signal_list(i) = numel(find(s20_TP_list == kpc_i));
end
s20_prop_correct_pred_signal = s20_N_correct_pred_signal_list; % percentage of TP for s20

% SNR decay calculation
dist_data = sort(kpc_dist);
s15_SFHo_snr01_data = s15_SFHo_snr01 .* (1./dist_data)./(1/0.1);
s20_SFHo_snr01_data = s20_SFHo_snr01 .* (1./dist_data)./(1/0.1);

figure, subplot(1,2,1), semilogx(s15_SFHo_snr01_data, s15_prop_correct_pred_signal, 'bo', 'MarkerFaceColor', 'auto'), grid on, ylabel("Correct Signal Classification [\%]"), xlabel("SNR");
title("Test Set Accuracy s15.0--SFHo"), ylim([0 100]), set(gca, 'XDir','reverse')
subplot(1,2,2), semilogx(s20_SFHo_snr01_data, s20_prop_correct_pred_signal, 'ro', 'MarkerFaceColor', 'auto'), grid on, ylabel("Correct Signal Classification [\%]"), xlabel("SNR");
title("Test Set Accuracy s20.0--SFHo"), ylim([0 100]), set(gca, 'XDir','reverse')
% visualize SNR decay as distance increase
figure, semilogy(dist_data, s15_SFHo_snr01_data, 'b*'), title("SNR Decay", 'Interpreter', 'latex'), xlabel("Distance [kpc]"), ylabel("SNR"), grid on, hold on
semilogy(dist_data, s20_SFHo_snr01_data, 'r*'), legend("s15.0--SFHo", "s20.0--SFHo")
%%
figure
cm = confusionchart(imds_test.Labels, predLabels, "ColumnSummary", "column-normalized", "RowSummary","row-normalized");
%%
% calculate statistics
TN = cm.NormalizedValues(1,1);
TP = cm.NormalizedValues(2,2);
FP = cm.NormalizedValues(1,2);
FN = cm.NormalizedValues(2,1);

accuracy = (TN+TP)/(TN+TP+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1_score = 2*precision*recall/(precision+recall)
TP_list = [];
% print the misclassified samples
for n = 1:N_test
    if (imds_test.Labels(n) ~= "noise")
        if (predLabels(n) == "signal")
            TP_list = [TP_list 1];
        else
            TP_list = [TP_list 0];
        end
    end
    
    if (imds_test.Labels(n) ~= predLabels(n))
        [~,name,~] = fileparts(imds_test.Files{n});
        fprintf("%s, at n = %i\n",name,n);
    end
end
%%
% ROC plot
rocObj = rocmetrics(imds_test.Labels, scoresList(:,2), "signal");
X = rocObj.Metrics.FalsePositiveRate;
Y = rocObj.Metrics.TruePositiveRate;
T = rocObj.Metrics.Threshold;
% threshold vs. false positive
figure, plot(T, Y-X), grid on, title("$T$ vs. TPR $-$ FAR"), hold on
xlabel("Positive Class Threshold $T$"), ylabel("True Positive Rate - False Alarm Rate")
[value, at] = max(Y-X);
opt_thres = T(at);
yline(value, '--'), xline(opt_thres, '--'), plot(opt_thres, value, 'o')
idx_model = find(T>=threshold,1,"last");
modelpt = [T(idx_model) X(idx_model) Y(idx_model)]
test_labels_count = countcats(imds_test.Labels);
p = test_labels_count(2);
n = test_labels_count(1);
cost = rocObj.Cost;
m = (cost(2,1)-cost(2,2))/(cost(1,2)-cost(1,1))*n/p;
[~,idx_opt] = min(X - Y/m);
optpt = [T(idx_opt) X(idx_opt) Y(idx_opt)] % optimal operating point
figure
r = plot(rocObj,ClassNames="signal");
hold on, grid on
scatter(optpt(2),optpt(3),"filled", ...
    DisplayName="Optimal Operating Point");
%%
idx_noise = imds_test.Labels == "noise";
idx_signal = imds_test.Labels == "signal";
bin_edges = 0:0.05:1;
[N_noise,~] = histcounts(scoresList(idx_noise,2),bin_edges);
[N_signal,~] = histcounts(scoresList(idx_signal,2),bin_edges);
figure, bar(bin_edges(2:end), [N_noise;N_signal], "grouped"), grid on
xline(threshold,'k--',"decision boundary")
legend("Noise","SNe Signal", "Location","best")
xlabel("$P_s$"), ylabel("Count")
title("Test Set Histogram of $P_s$")
%%
% GradCAM visualization
small_prob_idx = find(imds_test.Labels == "signal" & scoresList(:,2) <= 0.1);

% summary statistics about scoresList(:,2) < 0.1
[~,all_names,~] = fileparts(imds_test.Files(small_prob_idx));
dist_str = extractBefore(all_names, "kpc");
dist_str = extractAfter(dist_str, "_");
dist_small_prob = str2double(dist_str);
bins = unique(dist_small_prob);
dist_relative_occr = zeros(length(bins), 1);
for i = 1:length(bins)
    bin = bins(i);
    dist_relative_occr(i) = numel(find(dist_small_prob == bin));
end
figure, bar(bins, dist_relative_occr)
title("$P_s \le 0.1$ from Positive Class"), xlabel("Distance [kpc]"), ylabel("Relative Occurance [\%]"), grid on

i = randi(length(small_prob_idx));
idx = small_prob_idx(i);

[~,name,~] = fileparts(augimds_test.Files{idx});
fprintf("%s\n", name)
disp(scoresList(idx,:))
%%
% img_i = augimds_test.readByIndex(9);
% img_i = cat(3, img_i, img_i, img_i);

img_i = imread("C:\Users\gfd9111\OneDrive - AUT University\PUCV Supernova Project\PUCV-AUT-main\PUCV-AUT-main\Data\Simulate_GWs\new_simulation_training_qgram\signal\" + ...
    "s11.2--LS220_0.1kpc_sim1_SNR29.21.png");
[old_w, old_h] = size(img_i);

img_i = cat(3, img_i, img_i, img_i);
img_i = imresize(img_i, [224 224]);
[~,score] = classify(net, img_i)
if score(2) >= threshold
    pred_label = "signal";
else
    pred_label = "noise";
end

time_vect = linspace(0,2,old_w);
freq_vect = linspace(1600.2,562.7,old_h);

scoreMap = gradCAM(net,img_i,pred_label);
figure, subplot(1,2,1), imagesc(time_vect, freq_vect, img_i); title(sprintf("Prediction label: %s, True label: signal", pred_label)), colormap gray;
set(gca(), 'YDir', 'normal'), xlabel("Time [s]"), ylabel("Frequency [Hz]"), colorbar
subplot(1,2,2), imagesc(time_vect, freq_vect, rescale(scoreMap)), title("GradCAM Visualization")
colormap gray; colorbar
set(gca(), 'YDir', 'normal'), xlabel("Time [s]"), ylabel("Frequency [Hz]")
%%
% activation map
act1 = activations(net, img_i.input{:}, 'conv1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I), title("Activation Map"), colorbar
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,img_size(1:2));

I = imtile([img_i.input(:)',{act1chMax}]);
figure,imshow(I)
%%
function num_files = count_files_recursively(folder_path)
    % this function recursively search for the wavelet transform image at
    % each level, num_levels is an integer > 0

    % Get the list of text files and folders in the current directory
    contents = dir(folder_path);
    
    % Initialize counters
    num_files = 0;
    
    for i = 1:numel(contents)
        % Ignore "." and ".." entries
        if strcmp(contents(i).name, '.') || strcmp(contents(i).name, '..')
            continue;
        end
        
        % Full path to the current item
        item_path = fullfile(folder_path, contents(i).name);
        
        [~,~,ext] = fileparts(contents(i).name);
        % Check if the item is a file or a folder
        if contents(i).isdir
            % If it's a folder, recursively call the function
            num_files = num_files + count_files_recursively(item_path);
        elseif ext == ".txt"
            % If it's a text file, increment the counter
            num_files = num_files + 1;
        end
    end
end
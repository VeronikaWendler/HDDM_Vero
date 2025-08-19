%% Clean up the environment
%-----------------------------------------
clear all
clc
close all

m_num = 1
seed = sum(100*clock) + m_num + floor(1e6 * rand);
rng(seed);

%% Load task-related information
%-----------------------------------------

data = readtable('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv');
                 
data.OV = data.OV_2;
data.VD = data.VD_2;

% Convert 'cho' 1 = left, 2 = right into 'Choice' 0 = left, 1 = right
data.Choice = data.cho - 1;

% Rename 'corr' to 'Correct'
data.Correct = data.corr;

% **Remove rows with NaN values in the important columns**
cols_to_check = {'OV_2', 'VD_2', 'OV', 'VD', 'GazeDiff', 'FirstFixDur', 'FinalFixDur', 'MiddleFixDur', ...
                 'eachMiddleFixDur', 'GazeSwitch', 'FirstFixLoc', 'FinalFixLoc', ...
                 'DwellTimeAdvantage', 'chose_right'};

% Check that all specified columns exist in 'data' before using rmmissing
existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
cols_to_check = cols_to_check(existing_vars); % Keep only valid columns

data.OV = data.OV_2;
data.VD = data.VD_2;

% Apply rmmissing only to valid columns
data = rmmissing(data, 'DataVariables', cols_to_check);

% Filter only trials where phase is 'ES'
data = data(strcmp(data.phase, 'ES'), :);

% Define subject list (1 to 26 + 99), excluding specific subjects
subjlist = setdiff([1:26], [1, 4, 5, 6, 14, 99]);
nSubj = length(subjlist);
minSigma = randsample(0.02:0.001:0.03, nSubj, true); % HDDM assumes intra-trial variance is 1 for each time unit.


% Filter only trials where phase is 'ES'
data = data(strcmp(data.phase, 'ES'), :);

m_num = 1; % Model number
param_file = ['params_ES_m' num2str(m_num) '.csv'];
converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
conv_criteria = 1.1;

%% Load parameter sets
T = readtable(fullfile(param_file));
Tconv = readtable(fullfile(converging_file));

%% load individual parameter set
%-----------------------------------------
for k_subj = 1:nSubj
    clear paraname r
    subjID = subjlist(k_subj);
    paraname = {['a_subj.' num2str(subjID)], ...
        ['t_subj.' num2str(subjID)], ...
        ['v_Intercept_subj.' num2str(subjID)], ...
        ['v_AttentionW_subj.' num2str(subjID)], ...
        ['v_InattentionW_subj.' num2str(subjID)]
        };
    
    r = find(ismember(T.Var1, paraname));
    a = T.mean(r(1));
    ndt = T.mean(r(2));
    beta0 = T.mean(r(3));
    beta1 = T.mean(r(4));
    beta2 = T.mean(r(5));
    paramset(k_subj,:) = [a ndt beta0 beta1 beta2 minSigma(k_subj)];
    
        paraname_group_name = {'a', ...
        't', ...
        'v_Intercept', ...
        'v_AttentionW', ...
        'v_InattentionW'
        };

    r_group = find(ismember(T.Var1, paraname_group_name));
    
    a = T.mean(r_group(1));
    ndt = T.mean(r_group(2));
    beta0 = T.mean(r_group(3));
    beta1 = T.mean(r_group(4));
    beta2 = T.mean(r_group(5));
    theta(1,1) = beta2./beta1;
    paraname_group = [a ndt beta0 beta1 beta2 theta];
end
convergeset = Tconv.Gelman_Rubin;

save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_paramSim_fixedALL1','paramset','convergeset','paraname_group')

% Nrep = 100;

%% Start simulation
load('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_paramSim_fixedALL1')
TBsim = [];
h = waitbar(0, 'Please wait...');
for k_subj = 1:nSubj
    count = 0;
    temp_TBsim = [];
    subjID = subjlist(k_subj);
    LMH    = [1 2 3 4 5; ...
        1 2 3 4 5; ...
        1 2 3 4 5];

    waitbar(k_subj / nSubj)
    
    for k_OV = 1:3
        display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
        params = squeeze(paramset(k_subj,:));
        
        data.OV = data.OV_2;
        data.VD = data.VD_2;

        % Filter data for this subject & OV level
        data_subj = data(strcmp(data.phase, 'ES') & data.sub_id == subjID & data.OV == k_OV, {'phase', 'sub_id', 'VD', 'p1', 'p2', 'Correct', 'Choice'});
        values = [data_subj.p1, data_subj.p2].*1; % Check this ... Convert values to percentages


        for ktrial = 1:length(data_subj.sub_id)
            % Simulate aDDM process
            [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = EvidenceAccumulate_ES(k_OV, values(ktrial,1), values(ktrial,2), params);

            Correct = Choice == (values(ktrial,1) < values(ktrial,2));
            
            % Store simulation results
            count = count + 1;
            behData(count,:) = [subjID+1000 ktrial k_OV data_subj.VD(ktrial) values(ktrial,1),values(ktrial,2) values(ktrial,2)-values(ktrial,1) Choice Correct RT/1000];
            eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast,tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
                tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
        end
        
        varNames = {'sub_id';'trial';'OV';'VD';'Vl';'Vr';'RLdiff';'Choice';'Correct';'rt'; ...
            'Nfix';'FixLocFirst';'FixLocLast'; 'FixLocFirstCorr';'FixLocLastCorr'; 'DwellDiff'; ...
            'DwellFirst';'DwellMid';'DwellFinal';'eachDwellMiddle'};
        temp_TBsim = table(behData(:,1),behData(:,2),behData(:,3),behData(:,4),behData(:,5),behData(:,6), behData(:,7),behData(:,8),behData(:,9),behData(:,10), ...
            eyeData(:,1),eyeData(:,2),eyeData(:,3),eyeData(:,4),eyeData(:,5),eyeData(:,6), ...
            eyeData(:,7),eyeData(:,8),eyeData(:,9),eyeData(:,10),'VariableNames',varNames);
    end
    TBsim = [TBsim;temp_TBsim];
end

save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_paramSim_fixedALL1','TBsim','paramset','subjlist')
close(h)

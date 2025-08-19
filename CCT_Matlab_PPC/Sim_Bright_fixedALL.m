%% Clean up the environment
%-----------------------------------------
clear all
clc
close all
seed = rand; rng(seed);

%% Load task-related information
%-----------------------------------------
load('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Data/behav_organization.mat');
param_dir = '';
subjlist    = final_subjlist;
nSubj       = length(subjlist)
minSigma    = randsample(0.02:0.001:0.03,nSubj,true); % HDDM assumes the intra-trial variance is 1 for each time unit.

m_num = 1 % model1
    param_file= ['params_Bright_m' num2str(m_num) '.csv'];
    converging_file= ['gelman_rubin_m' num2str(m_num) '.csv'];
    conv_criteria = 1.1;
    %% read csv for parameters and converging document separately.
    %-----------------------------------------
    T = readtable(fullfile(param_dir, param_file));
    Tconv = readtable(fullfile(param_dir, converging_file));
    
    %% load individual parameter set
    %-----------------------------------------
    for k_subj = 1:nSubj
        clear paraname r
        subjID = subjlist(k_subj);
        paraname =  {['a_subj.' num2str(subjID)], ...
            ['t_subj.' num2str(subjID)], ...
            ['v_Intercept_subj.' num2str(subjID)], ...
            ['v_AttentionW_subj.' num2str(subjID)], ...
            ['v_InattentionW_subj.' num2str(subjID)]
            };
        
        r = find(ismember(T.Var1, paraname));
        a     = T.mean(r(1));
        ndt   = T.mean(r(2));
        beta0 = T.mean(r(3));
        beta1 = T.mean(r(4));
        beta2 = T.mean(r(5));
        paramset(k_subj,:) = [a ndt beta0 beta1 beta2 minSigma(k_subj)];
        
        % group parameters
        paraname_group_name =  {'a', ...
        't', ...
        'v_Intercept', ...
        'v_AttentionW', ...
        'v_InattentionW'
        };
    
    r_group = find(ismember(T.Var1, paraname_group_name));
    
    a     = T.mean(r_group(1));
    ndt   = T.mean(r_group(2));
    beta0 = T.mean(r_group(3));
    beta1 = T.mean(r_group(4));
    beta2 = T.mean(r_group(5));
    theta(1,1) = beta2./beta1;    
    paraname_group= [a ndt beta0 beta1 beta2 theta];
    end
    convergeset = Tconv.Gelman_Rubin;
    
  
    
    save('Sim_HDDM_Bright_paramSim_fixedALL','paramset','convergeset','paraname_group')

%% start simulation.
%-----------------------------------------
load('Sim_HDDM_Bright_paramSim_fixedALL')
TBsim = [];
k_model = 6; % the sixth model in the winning model
h = waitbar(0,'Please wait...');
for k_subj = 1:nSubj
    count = 0;
    temp_TBsim = [];
    subjID = subjlist(k_subj);
    LMH    = [1 2 5 6 7; ...
        1 3 5 6 8; ...
        1 4 5 6 9];
    
    % computations take place here
    waitbar(k_subj / nSubj)
    
    
    for k_OV = 1:3
        display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
        params = squeeze(paramset(k_subj,:));
        data  = TB.Bright(TB.Bright.SubjID== subjID & TB.Bright.OV==k_OV,:);
        %         values   = sort([data.leftOption data.rightOption],2,'descend'); % make sure Va is always the better option.
        values   = [data.leftOption data.rightOption].*100;
        
        for ktrial = 1:length(data.SubjID)
            
            % main aDDM simulation function
            [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur]= EvidenceAccumulate_Bright(k_OV,values(ktrial,1),values(ktrial,2),params);
            
            Correct  = Choice == (values(ktrial,1)<values(ktrial,2));
            
            
            % store in the file
            count = count+1;
            behData(count,:) = [subjID+1000 ktrial k_OV data.VD(ktrial) values(ktrial,1),values(ktrial,2) values(ktrial,2)-values(ktrial,1) Choice Correct RT/1000];
            eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast,tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
                tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
        end
        
        varNames = {'SubjID';'trial';'OV';'VD';'Vl';'Vr';'RLdiff';'Choice';'Correct';'rt'; ...
            'Nfix';'FixLocFirst';'FixLocLast'; 'FixLocFirstCorr';'FixLocLastCorr'; 'DwellDiff'; ...
            'DwellFirst';'DwellMid';'DwellFinal';'eachDwellMiddle'};
        temp_TBsim = table(behData(:,1),behData(:,2),behData(:,3),behData(:,4),behData(:,5),behData(:,6), behData(:,7),behData(:,8),behData(:,9),behData(:,10), ...
            eyeData(:,1),eyeData(:,2),eyeData(:,3),eyeData(:,4),eyeData(:,5),eyeData(:,6), ...
            eyeData(:,7),eyeData(:,8),eyeData(:,9),eyeData(:,10),'VariableNames',varNames);
    end
    TBsim = [TBsim;temp_TBsim];
end

save('sim_Bright_fixedALL','TBsim','paramset','final_subjlist')
close(h)
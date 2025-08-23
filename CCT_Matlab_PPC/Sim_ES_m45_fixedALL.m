% A modification of Dr Chih-Chung Tings code
% This code is for the dual addm (parameters for E and S)
% here is the formula:
%# original drift rate formula:
%# v = β0 + β1 * (PropDwell_opt​ * V_opt​ − PropDwell_sub * V_sub) + β2 * (PropDwell_sub * V_opt​ − PropDwell_opt​ * V_sub)+ϵ
%# dirft rate with two separate thetas for S and E :
%# v = β0 + β1 * AttentionW_E + β2 * AttentionW_S + β3 * InattentionW_E + β4 * InattentionW_S  +ϵ

%# where:
%# AttentionW_E = Value_E_opt * DwellProp_E - Value_S_sub * DwellProp_S
%# AttentionW_S = Value_S_opt * DwellProp_S - Value_E_sub * DwellProp_E
%# InattentionW_E = Value_E_opt * DwellProp_S - Value_S_sub * DwellProp_E
%# InattentionW_S = Value_S_opt * DwellProp_E - Value_E_sub * DwellProp_S

%#where:

%# Value_E_opt = value of E-option when E option > S option on that trial
%# Value_E_sub = value of E-option when E option < S option on that trial
%# Value_S_sub = value of S-option when S option < E option on that trial
%# Value_S_opt = value of S-option when S option > E option on that trial
%# DwellProp_E = proportion of dwell time on E option
%# DwellProp_S = proportion of dwell time on S option

%% Clean up the environment
%-----------------------------------------
clear all
clc
close all

m_num = 45
seed = sum(100*clock) + m_num + floor(1e6 * rand);
rng(seed);

%% Load task-related information

%-----------------------------------------

data = readtable('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/data/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv');
                 
data.OV = data.OV_2;
data.VD = data.VD_2;

% Convert 'cho' 1 = left, 2 = right into 'Choice' 0 = left, 1 = right - this is the equivalent of the chose_right column so chose_right can also be used (here it really depends on what we want choice to be)
data.Choice = data.cho - 1;

% this model is accuracy coded 
% Rename 'corr' to 'Correct'
data.Correct = data.corr;

% cleaning
cols_to_check = {'OV_2', 'VD_2', 'OV', 'VD', 'GazeDiff', 'FirstFixDur', 'FinalFixDur', 'MiddleFixDur', ...
                 'eachMiddleFixDur', 'GazeSwitch', 'FirstFixLoc', 'FinalFixLoc', ...
                 'DwellTimeAdvantage', 'chose_right'};


existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
cols_to_check = cols_to_check(existing_vars); 
data.OV = data.OV_2;
data.VD = data.VD_2;

data = rmmissing(data, 'DataVariables', cols_to_check);

data = data(strcmp(data.phase, 'ES'), :);

% excluding specific subjects
subjlist = setdiff([1:26], [1, 4, 5, 6, 14, 99]);
nSubj = length(subjlist);
minSigma = randsample(0.02:0.001:0.03, nSubj, true); % HDDM assumes intra-trial variance is 1 for each time unit.


% Filter only trials where phase is 'ES'
data = data(strcmp(data.phase, 'ES'), :);

m_num = 45; % Model number
param_file = ['params_ES_m' num2str(m_num) '.csv'];
converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
conv_criteria = 1.1;


T = readtable(fullfile(param_file));
Tconv = readtable(fullfile(converging_file));

%individual parameter set

for k_subj = 1:nSubj
    subjID = subjlist(k_subj);
    paraname = {['a_subj.' num2str(subjID)], ...
                ['t_subj.' num2str(subjID)], ...
                ['v_Intercept_subj.' num2str(subjID)], ...
                ['v_AttentionW_E_subj.' num2str(subjID)], ...
                ['v_AttentionW_S_subj.' num2str(subjID)], ...
                ['v_InattentionW_E_subj.' num2str(subjID)], ...
                ['v_InattentionW_S_subj.' num2str(subjID)]};
    r = find(ismember(T.Var1, paraname));

    a     = T.mean(r(1));
    ndt   = T.mean(r(2));
    beta0 = T.mean(r(3));
    beta1 = T.mean(r(4));
    beta2 = T.mean(r(5));
    beta3 = T.mean(r(6));
    beta4 = T.mean(r(7));

    % You used 8 entries before; keep same layout with minSigma at end:
    paramset(k_subj,:) = [a ndt beta0 beta1 beta2 beta3 beta4 minSigma(k_subj)];
end

% group-level
paraname_group_name = {'a','t','v_Intercept','v_AttentionW_E','v_AttentionW_S','v_InattentionW_E','v_InattentionW_S'};
r_group = find(ismember(T.Var1, paraname_group_name));
a     = T.mean(r_group(1));
ndt   = T.mean(r_group(2));
beta0 = T.mean(r_group(3));
beta1 = T.mean(r_group(4));
beta2 = T.mean(r_group(5));
beta3 = T.mean(r_group(6));
beta4 = T.mean(r_group(7));
theta_E(1,1) = beta3./beta1;
theta_S(1,1) = beta4./beta2;
paraname_group = [a ndt beta0 beta1 beta2 beta3 beta4 theta_E theta_S];

convergeset = Tconv.Gelman_Rubin;

save('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/sim/Sim_HDDM_ES_m45_paramSim_fixedALL','paramset','convergeset','paraname_group')


%% Start simulation
load('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/sim/Sim_HDDM_ES_m45_paramSim_fixedALL')
TBsim = [];
h = waitbar(0, 'Please wait...');
for k_subj = 1:nSubj
    count = 0;
    temp_TBsim = [];
    subjID = subjlist(k_subj);
    LMH    = [1 4 5 6 7 8 9; ...
        2 4 5 6 7 8 9; ...
        3 4 5 6 7 8 9];

    waitbar(k_subj / nSubj)
    
    for k_OV = 1:3
        display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
        params = squeeze(paramset(k_subj,:));
        
        data.OV = data.OV_2;
        data.VD = data.VD_2;

        % Filter data for this subject & OV level
        data_subj = data(strcmp(data.phase, 'ES') & data.sub_id == subjID & data.OV == k_OV, {'phase', 'sub_id', 'VD', 'p1', 'p2', 'Correct', 'Choice'});
        values = [data_subj.p1, data_subj.p2].*1; % Check this ... Convert values to percentages, in my code they are already percentages


        for ktrial = 1:length(data_subj.sub_id)
            % Simulate aDDM process
            [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = EvidenceAccumulate_ES_m45(k_OV, values(ktrial,1), values(ktrial,2), params);

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

save('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/sim/without_Fix_weight_FixR/sim_ES_m45_fixedALL','TBsim','paramset','subjlist')
close(h)





















% 
% 
% %% Simulate ES with empirical-like fixation sequences & durations
% % Uses subject×OV fixation metrics from make_fixation_metrics_ES.m
% % and passes FixR_seq / FixDur_seq into EvidenceAccumulate_ES_m45.
% 
% %-----------------------------------------
% clear all; clc; close all;
% 
% m_num = 45;
% seed = sum(100*clock) + m_num + floor(1e6 * rand);
% rng(seed);
% 
% %% Paths (adjust if needed)
% data_file = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv';
% fix_file  = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/fixation_metrics_ES.mat';
% 
% param_file      = ['params_ES_m' num2str(m_num) '.csv'];
% converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
% 
% save_paramset_to = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m45_paramSim_fixedALL_empFix';
% save_sim_to      = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/sim_ES_m45_empFix';
% 
% %% Load task-related data
% data = readtable(data_file);
% 
% % keep ES, basic cleaning like your original
% data.OV = data.OV_2;
% data.VD = data.VD_2;
% data.Choice  = data.cho - 1;
% data.Correct = data.corr;
% 
% cols_to_check = {'OV_2','VD_2','OV','VD','GazeDiff','FirstFixDur','FinalFixDur','MiddleFixDur', ...
%                  'eachMiddleFixDur','GazeSwitch','FirstFixLoc','FinalFixLoc','DwellTimeAdvantage','chose_right'};
% existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
% cols_to_check = cols_to_check(existing_vars);
% 
% data = rmmissing(data, 'DataVariables', cols_to_check);
% data = data(strcmp(data.phase,'ES'), :);
% 
% % subject list
% subjlist = setdiff(1:26, [1,4,5,6,14,99]);   % exclude poor subs
% nSubj = numel(subjlist);
% minSigma = randsample(0.02:0.001:0.03, nSubj, true);
% 
% %% Load fixation metrics (built by make_fixation_metrics_ES.m)
% load(fix_file, 'fixm');                         % contains 'fixm'
% sub_ids   = fixm.sub_ids;                       % subject ids present in metrics
% ov_labels = fixm.ov_order;                      % {'low','medium','high'}
% 
% %% Load HDDM parameter sets
% T    = readtable(fullfile(param_file));
% Tconv= readtable(fullfile(converging_file));
% 
% % Build per-subject parameter vectors
% paramset = nan(nSubj, 8);
% theta_E = nan(1,1); theta_S = nan(1,1); % kept for compatibility with your save
% 
% for k_subj = 1:nSubj
%     subjID = subjlist(k_subj);
%     paraname = {['a_subj.' num2str(subjID)], ...
%                 ['t_subj.' num2str(subjID)], ...
%                 ['v_Intercept_subj.' num2str(subjID)], ...
%                 ['v_AttentionW_E_subj.' num2str(subjID)], ...
%                 ['v_AttentionW_S_subj.' num2str(subjID)], ...
%                 ['v_InattentionW_E_subj.' num2str(subjID)], ...
%                 ['v_InattentionW_S_subj.' num2str(subjID)]};
%     r = find(ismember(T.Var1, paraname));
% 
%     a     = T.mean(r(1));
%     ndt   = T.mean(r(2));
%     beta0 = T.mean(r(3));
%     beta1 = T.mean(r(4));
%     beta2 = T.mean(r(5));
%     beta3 = T.mean(r(6));
%     beta4 = T.mean(r(7));
% 
%     % You used 8 entries before; keep same layout with minSigma at end:
%     paramset(k_subj,:) = [a ndt beta0 beta1 beta2 beta3 beta4 minSigma(k_subj)];
% end
% 
% % group-level (optional; you saved it before, keep parity)
% paraname_group_name = {'a','t','v_Intercept','v_AttentionW_E','v_AttentionW_S','v_InattentionW_E','v_InattentionW_S'};
% r_group = find(ismember(T.Var1, paraname_group_name));
% a     = T.mean(r_group(1));
% ndt   = T.mean(r_group(2));
% beta0 = T.mean(r_group(3));
% beta1 = T.mean(r_group(4));
% beta2 = T.mean(r_group(5));
% beta3 = T.mean(r_group(6));
% beta4 = T.mean(r_group(7));
% theta_E(1,1) = beta3./beta1;
% theta_S(1,1) = beta4./beta2;
% paraname_group = [a ndt beta0 beta1 beta2 beta3 beta4 theta_E theta_S];
% 
% convergeset = Tconv.Gelman_Rubin;
% 
% % save paramset (like you did)
% save([save_paramset_to '_o_noOVfixR'], 'paramset','convergeset','paraname_group');
% 
% %% --- Simulation ---
% TBsim = [];
% h = waitbar(0, 'Simulating...');
% 
% for k_subj = 1:nSubj
%     waitbar(k_subj / nSubj, h);
%     subjID = subjlist(k_subj);
%     params = squeeze(paramset(k_subj,:));
% 
%     % for clarity, your OV levels in data are 1/2/3; map to labels
%     % ov_labels = {'low','medium','high'}; so 1->'low', 2->'medium', 3->'high'
%     for k_OV = 1:3
%         fprintf('subj %d, OV %d\n', subjID, k_OV);
% 
%         % pull fixation metrics (subject-level with group fallback)
%         ov_label = ov_labels{k_OV};
%         si = find(sub_ids == subjID, 1, 'first');
%         oi = find(strcmp(ov_labels, ov_label), 1, 'first');
% 
%         if isempty(oi)
%             error('OV label not found in fixm.ov_order');
%         end
% 
%         if isempty(si)
%             % subject not in fixm (should not happen), use group only
%             warning('Subject %d not found in fixation metrics; using group fallback only.', subjID);
%             S = fixm.groupOV(oi);  % fake as "subject" struct
%             G = fixm.groupOV(oi);
%         else
%             S = fixm.by_subjOV(si, oi);
%             G = fixm.groupOV(oi);
%         end
% 
%         p_firstR_VrGt = pick(S.p_firstR_VrGt, G.p_firstR_VrGt, 0.5);
%         p_firstR_VlGt = pick(S.p_firstR_VlGt, G.p_firstR_VlGt, 0.5);
%         p_firstR_Eq   = pick(S.p_firstR_Eq,   G.p_firstR_Eq,   0.5);
%         p_switch      = pick(S.p_switch,      G.p_switch,      0.5);
% 
%         [nfix_vals_s, nfix_pmf_s] = get_pmf(S);
%         if isempty(nfix_vals_s)
%             nfix_vals_s = G.Nfix_values;
%             nfix_pmf_s  = G.Nfix_pmf;
%         end
% 
%         ln_mu_first   = pick(S.ln_mu_first,    G.ln_mu_first,    log(250));
%         ln_sg_first   = pick(S.ln_sigma_first, G.ln_sigma_first, 0.4);
%         ln_mu_middle  = pick(S.ln_mu_middle,   G.ln_mu_middle,   log(250));
%         ln_sg_middle  = pick(S.ln_sigma_middle,G.ln_sigma_middle,0.4);
%         ln_mu_final   = pick(S.ln_mu_final,    G.ln_mu_final,    log(250));
%         ln_sg_final   = pick(S.ln_sigma_final, G.ln_sigma_final, 0.4);
% 
%         % subject×OV trials (values etc.)
%         data_subj = data(data.sub_id==subjID & data.OV==k_OV, {'phase','sub_id','VD','p1','p2','Correct','Choice'});
%         values    = [data_subj.p1, data_subj.p2];   % your model expects original scale (you had .*1)
% 
%         % collect simulated rows
%         count = 0; clear behData eyeData
% 
%         for ktrial = 1:height(data_subj)
%             Vl = values(ktrial,1);
%             Vr = values(ktrial,2);
% 
%             % sample Nfix from pmf (can switch to empirical Nfix if you add that column)
%             Nfix = sample_from_pmf(nfix_vals_s, nfix_pmf_s);
% 
%             % sequence & durations from empirical-like generators
%             FixR_seq   = simulate_fixation_sequence(Vl, Vr, Nfix, ...
%                             p_firstR_VrGt, p_firstR_VlGt, p_firstR_Eq, p_switch);
% 
%             FixDur_seq = sample_fixation_durations_LN(Nfix, ...
%                             ln_mu_first, ln_sg_first, ln_mu_middle, ln_sg_middle, ln_mu_final, ln_sg_final);
% 
%             % run your accumulator (updated to require sequences)
%             [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = ...
%                 EvidenceAccumulate_ES_m45(k_OV, Vl, Vr, params, FixR_seq, FixDur_seq);
% 
%             % store
%             count = count + 1;
%             behData(count,:) = [subjID+1000, ktrial, k_OV, data_subj.VD(ktrial), ...
%                                 Vl, Vr, (Vr-Vl), Choice, (Choice == (Vl<Vr)), RT/1000];
%             eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast, ...
%                                 tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
%                                 tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
%         end
% 
%         % pack as table (as you did)
%         varNames = {'sub_id','trial','OV','VD','Vl','Vr','RLdiff','Choice','Correct','rt', ...
%                     'Nfix','FixLocFirst','FixLocLast','FixLocFirstCorr','FixLocLastCorr','DwellDiff', ...
%                     'DwellFirst','DwellMid','DwellFinal','eachDwellMiddle'};
% 
%         temp_TBsim = table(behData(:,1),behData(:,2),behData(:,3),behData(:,4),behData(:,5),behData(:,6),behData(:,7), ...
%                            behData(:,8),behData(:,9),behData(:,10), ...
%                            eyeData(:,1),eyeData(:,2),eyeData(:,3),eyeData(:,4),eyeData(:,5),eyeData(:,6), ...
%                            eyeData(:,7),eyeData(:,8),eyeData(:,9),eyeData(:,10), ...
%                            'VariableNames', varNames);
% 
%         TBsim = [TBsim; temp_TBsim];
%     end
% end
% 
% close(h);
% 
% save([save_sim_to '_noOVfixR'], 'TBsim','paramset','subjlist');
% 
% %% ----------------- Helper functions (local) -----------------
% function v = pick(x, fallback, def)
%     if ~isfinite(x), x = NaN; end
%     if isnan(x) || isempty(x)
%         if ~isfinite(fallback) || isempty(fallback)
%             v = def;
%         else
%             v = fallback;
%         end
%     else
%         v = x;
%     end
% end
% 
% function [vals, pmf] = get_pmf(S)
%     vals = []; pmf = [];
%     if isfield(S,'Nfix_values') && isfield(S,'Nfix_pmf')
%         vals = S.Nfix_values; pmf = S.Nfix_pmf;
%         if isempty(vals) || isempty(pmf)
%             vals = []; pmf = [];
%         end
%     end
% end
% 
% function N = sample_from_pmf(vals, pmf)
%     if isempty(vals)
%         N = 1; return
%     end
%     cs = cumsum(pmf(:)');
%     r = rand;
%     idx = find(r <= cs, 1, 'first');
%     if isempty(idx), idx = numel(vals); end
%     N = vals(idx);
% end
% 
% function FixR_seq = simulate_fixation_sequence(Vl, Vr, Nfix, p_firstR_VrGt, p_firstR_VlGt, p_firstR_Eq, p_switch)
%     if Nfix < 1, FixR_seq = []; return; end
%     if Vr > Vl
%         p0 = p_firstR_VrGt;
%     elseif Vl > Vr
%         p0 = p_firstR_VlGt;
%     else
%         p0 = p_firstR_Eq;
%     end
%     FixR_seq = zeros(1, Nfix);
%     FixR_seq(1) = rand < p0;
% 
%     ps = max(min(p_switch, 0.999), 0.001); % keep in (0,1)
%     for t = 2:Nfix
%         if rand < ps
%             FixR_seq(t) = 1 - FixR_seq(t-1); % switch side
%         else
%             FixR_seq(t) = FixR_seq(t-1);     % stay
%         end
%     end
% end
% 
% function FixDur_seq = sample_fixation_durations_LN(Nfix, muF, sgF, muM, sgM, muL, sgL)
%     if Nfix < 1
%         FixDur_seq = [];
%         return
%     end
%     drawLN = @(mu,sg,n) max(1, round(lognrnd(mu, sg, [1,n])));
%     if Nfix == 1
%         FixDur_seq = drawLN(muF, sgF, 1);
%     elseif Nfix == 2
%         FixDur_seq = [drawLN(muF, sgF, 1), drawLN(muL, sgL, 1)];
%     else
%         mids = drawLN(muM, sgM, Nfix-2);
%         FixDur_seq = [drawLN(muF, sgF, 1), mids, drawLN(muL, sgL, 1)];
%     end
% end
% 
% 















































% %% Clean up the environment
% %-----------------------------------------
% clear all
% clc
% close all
% 
% m_num = 45
% seed = sum(100*clock) + m_num + floor(1e6 * rand);
% rng(seed);
% 
% %% Load task-related information
% %-----------------------------------------
% 
% data = readtable('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv');
% 
% 
% fix_file = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/fixation_metrics_ES.mat';
% load(fix_file, 'fixm');
% sub_ids = fixm.sub_ids;
% ov_labels = fixm.ov_order;  % {'low','medium','high'}
% 
% 
% 
% 
% data.OV = data.OV_2;
% data.VD = data.VD_2;
% 
% % Convert 'cho' 1 = left, 2 = right into 'Choice' 0 = left, 1 = right
% data.Choice = data.cho - 1;
% 
% % Rename 'corr' to 'Correct'
% data.Correct = data.corr;
% 
% % **Remove rows with NaN values in the important columns**
% cols_to_check = {'OV_2', 'VD_2', 'OV', 'VD', 'GazeDiff', 'FirstFixDur', 'FinalFixDur', 'MiddleFixDur', ...
%                  'eachMiddleFixDur', 'GazeSwitch', 'FirstFixLoc', 'FinalFixLoc', ...
%                  'DwellTimeAdvantage', 'chose_right'};
% 
% % Check that all specified columns exist in 'data' before using rmmissing
% existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
% cols_to_check = cols_to_check(existing_vars); % Keep only valid columns
% 
% data.OV = data.OV_2;
% data.VD = data.VD_2;
% 
% % Apply rmmissing only to valid columns
% data = rmmissing(data, 'DataVariables', cols_to_check);
% 
% % Filter only trials where phase is 'ES'
% data = data(strcmp(data.phase, 'ES'), :);
% 
% % Define subject list (1 to 26 + 99), excluding specific subjects
% subjlist = setdiff([1:26], [1, 4, 5, 6, 14, 99]); % excluding poor subs   
% nSubj = length(subjlist);
% minSigma = randsample(0.02:0.001:0.03, nSubj, true); % HDDM assumes intra-trial variance is 1 for each time unit.
% 
% 
% % Filter only trials where phase is 'ES'
% data = data(strcmp(data.phase, 'ES'), :);
% 
% m_num = 45; % Model number
% param_file = ['params_ES_m' num2str(m_num) '.csv'];
% converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
% conv_criteria = 1.1;
% 
% %% Load parameter sets
% T = readtable(fullfile(param_file));
% Tconv = readtable(fullfile(converging_file));
% 
% %% load individual parameter set
% %-----------------------------------------
% for k_subj = 1:nSubj
%     clear paraname r
%     subjID = subjlist(k_subj);
%     paraname = {['a_subj.' num2str(subjID)], ...
%         ['t_subj.' num2str(subjID)], ...
%         ['v_Intercept_subj.' num2str(subjID)], ...
%         ['v_AttentionW_E_subj.' num2str(subjID)], ...
%         ['v_AttentionW_S_subj.' num2str(subjID)], ...
%         ['v_InattentionW_E_subj.' num2str(subjID)], ...
%         ['v_InattentionW_S_subj.' num2str(subjID)]
%         };
%     
%     r = find(ismember(T.Var1, paraname));
%     a = T.mean(r(1));
%     ndt = T.mean(r(2));
%     beta0 = T.mean(r(3));
%     beta1 = T.mean(r(4));
%     beta2 = T.mean(r(5));
%     beta3 = T.mean(r(6));
%     beta4 = T.mean(r(7));
%     paramset(k_subj,:) = [a ndt beta0 beta1 beta2 beta3 beta4 minSigma(k_subj)];
%     
%         paraname_group_name = {
%         'a', ...
%         't', ...
%         'v_Intercept', ...
%         'v_AttentionW_E', ...
%         'v_AttentionW_S', ...
%         'v_InattentionW_E', ...
%         'v_InattentionW_S', ...
%         };
% 
%     r_group = find(ismember(T.Var1, paraname_group_name));
%     
%     a = T.mean(r_group(1));
%     ndt = T.mean(r_group(2));
%     beta0 = T.mean(r_group(3));
%     beta1 = T.mean(r_group(4));
%     beta2 = T.mean(r_group(5));
%     beta3 = T.mean(r_group(6));
%     beta4 = T.mean(r_group(7));
%     theta_E(1,1) = beta3./beta1;
%     theta_S(1,1) = beta4./beta2;
% 
%     paraname_group = [a ndt beta0 beta1 beta2 beta3 beta4 theta_E theta_S];
% end
% convergeset = Tconv.Gelman_Rubin;
% 
% save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m45_paramSim_fixedALL_o_noOVfixR','paramset','convergeset','paraname_group')
% 
% 
% %% Start simulation
% load('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m45_paramSim_fixedALL_o_noOVfixR')
% TBsim = [];
% h = waitbar(0, 'Please wait...');
% for k_subj = 1:nSubj
%     count = 0;
%     temp_TBsim = [];
%     subjID = subjlist(k_subj);
%     LMH    = [1 2 3 4 5 6 7; ...
%         1 2 3 4 5 6 7; ...
%         1 2 3 4 5 6 7];
% 
%     waitbar(k_subj / nSubj)
%     
%     for k_OV = 1:3
%         display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
%         params = squeeze(paramset(k_subj,:));
%         
%         data.OV = data.OV_2;
%         data.VD = data.VD_2;
% 
%         % Filter data for this subject & OV level
%         data_subj = data(strcmp(data.phase, 'ES') & data.sub_id == subjID & data.OV == k_OV, {'phase', 'sub_id', 'VD', 'p1', 'p2', 'Correct', 'Choice'});
%         values = [data_subj.p1, data_subj.p2].*1; % Check this ... Convert values to percentages, in my code they are already percentages
% 
% 
%         for ktrial = 1:length(data_subj.sub_id)
%             % Simulate aDDM process
%             [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = EvidenceAccumulate_ES_m45(k_OV, values(ktrial,1), values(ktrial,2), params);
% 
%             Correct = Choice == (values(ktrial,1) < values(ktrial,2));
%             
%             % Store simulation results
%             count = count + 1;
%             behData(count,:) = [subjID+1000 ktrial k_OV data_subj.VD(ktrial) values(ktrial,1),values(ktrial,2) values(ktrial,2)-values(ktrial,1) Choice Correct RT/1000];
%             eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast,tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
%                 tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
%         end
%         
%         varNames = {'sub_id';'trial';'OV';'VD';'Vl';'Vr';'RLdiff';'Choice';'Correct';'rt'; ...
%             'Nfix';'FixLocFirst';'FixLocLast'; 'FixLocFirstCorr';'FixLocLastCorr'; 'DwellDiff'; ...
%             'DwellFirst';'DwellMid';'DwellFinal';'eachDwellMiddle'};
%         temp_TBsim = table(behData(:,1),behData(:,2),behData(:,3),behData(:,4),behData(:,5),behData(:,6), behData(:,7),behData(:,8),behData(:,9),behData(:,10), ...
%             eyeData(:,1),eyeData(:,2),eyeData(:,3),eyeData(:,4),eyeData(:,5),eyeData(:,6), ...
%             eyeData(:,7),eyeData(:,8),eyeData(:,9),eyeData(:,10),'VariableNames',varNames);
%     end
%     TBsim = [TBsim;temp_TBsim];
% end
% 
% save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/sim_ES_m45_fixedALL_noOVfixR','TBsim','paramset','subjlist')
% close(h)


%         for ktrial = 1:length(data_subj.sub_id)
%             % Simulate aDDM process
%             [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = EvidenceAccumulate_ES(k_OV, values(ktrial,1), values(ktrial,2), params);
% 
%             Correct = Choice == (values(ktrial,1) < values(ktrial,2));
%             
%             % Store simulation results
%             count = count + 1;
%             behData(count,:) = [subjID+1000 ktrial k_OV data_subj.VD(ktrial) values(ktrial,1),values(ktrial,2) values(ktrial,2)-values(ktrial,1) Choice Correct RT/1000];
%             eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast,tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
%                 tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
%         end


% // m_num = 1 % model1
% //     param_file= ['params_ES_m' num2str(m_num) '.csv'];
% //     converging_file= ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
% //     conv_criteria = 1.1;
% //     %% read csv for parameters and converging document separately.
% //     %-----------------------------------------
% //     T = readtable(fullfile(param_dir, param_file));
% //     Tconv = readtable(fullfile(param_dir, converging_file));
%     
% //     %% load individual parameter set
% //     %-----------------------------------------
% //     for k_subj = 1:nSubj
% //         clear paraname r
% //         subjID = subjlist(k_subj);
% //         paraname =  {['a_subj.' num2str(subjID)], ...
% //             ['t_subj.' num2str(subjID)], ...
% //             ['v_Intercept_subj.' num2str(subjID)], ...
% //             ['v_AttentionW_subj.' num2str(subjID)], ...
% //             ['v_InattentionW_subj.' num2str(subjID)]
% //             };
%         
% //         r = find(ismember(T.Var1, paraname));
% //         a     = T.mean(r(1));
% //         ndt   = T.mean(r(2));
% //         beta0 = T.mean(r(3));
% //         beta1 = T.mean(r(4));
% //         beta2 = T.mean(r(5));
% //         paramset(k_subj,:) = [a ndt beta0 beta1 beta2 minSigma(k_subj)];
%         
% //         % group parameters
% //         paraname_group_name =  {'a', ...
% //         't', ...
% //         'v_Intercept', ...
% //         'v_AttentionW', ...
% //         'v_InattentionW'
% //         };
%     
% //     r_group = find(ismember(T.Var1, paraname_group_name));
%     
% //     a     = T.mean(r_group(1));
% //     ndt   = T.mean(r_group(2));
% //     beta0 = T.mean(r_group(3));
% //     beta1 = T.mean(r_group(4));
% //     beta2 = T.mean(r_group(5));
% //     theta(1,1) = beta2./beta1;    
% //     paraname_group= [a ndt beta0 beta1 beta2 theta];
% //     end
% //     convergeset = Tconv.Gelman_Rubin;
%     
%   
%     
% //     save('Sim_HDDM_ES_paramSim_fixedALL','paramset','convergeset','paraname_group')
% 
% // %% start simulation.
% // %-----------------------------------------
% // load('Sim_HDDM_ES_paramSim_fixedALL')
% // TBsim = [];
% // k_model = 6; % the sixth model in the winning model
% // h = waitbar(0,'Please wait...');
% // for k_subj = 1:nSubj
% //     count = 0;
% //     temp_TBsim = [];
% //     subjID = subjlist(k_subj);
% //     LMH    = [1 2 5 6 7; ...
% //         1 3 5 6 8; ...
% //         1 4 5 6 9];
%     
% //     % computations take place here
% //     waitbar(k_subj / nSubj)
%     
%     
% //     for k_OV = 1:3
% //         display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
% //         params = squeeze(paramset(k_subj,:));
% //         data  = TB.Bright(TB.Bright.SubjID== subjID & TB.Bright.OV==k_OV,:);
% //         %         values   = sort([data.leftOption data.rightOption],2,'descend'); % make sure Va is always the better option.
% //         values   = [data.leftOption data.rightOption].*100;
%         
% //         for ktrial = 1:length(data.SubjID)
%             
% //             % main aDDM simulation function
% //             [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur]= EvidenceAccumulate_Bright(k_OV,values(ktrial,1),values(ktrial,2),params);
%             
% //             Correct  = Choice == (values(ktrial,1)<values(ktrial,2));
%             
%             
% //             % store in the file
% //             count = count+1;
% //             behData(count,:) = [subjID+1000 ktrial k_OV data.VD(ktrial) values(ktrial,1),values(ktrial,2) values(ktrial,2)-values(ktrial,1) Choice Correct RT/1000];
% //             eyeData(count,:) = [tempEyeData.Nfix, tempEyeData.FixLocFirst, tempEyeData.FixLocLast,tempEyeData.FixLocFirstCorr, tempEyeData.FixLocLastCorr, tempEyeData.DwellDiff, ...
% //                 tempEyeData.FirstFixDur, tempEyeData.MiddleFixDur, tempEyeData.FinalFixDur, tempEyeData.eachMiddleFixDur];
% //         end
%         
% //         varNames = {'SubjID';'trial';'OV';'VD';'Vl';'Vr';'RLdiff';'Choice';'Correct';'rt'; ...
% //             'Nfix';'FixLocFirst';'FixLocLast'; 'FixLocFirstCorr';'FixLocLastCorr'; 'DwellDiff'; ...
% //             'DwellFirst';'DwellMid';'DwellFinal';'eachDwellMiddle'};
% //         temp_TBsim = table(behData(:,1),behData(:,2),behData(:,3),behData(:,4),behData(:,5),behData(:,6), behData(:,7),behData(:,8),behData(:,9),behData(:,10), ...
% //             eyeData(:,1),eyeData(:,2),eyeData(:,3),eyeData(:,4),eyeData(:,5),eyeData(:,6), ...
% //             eyeData(:,7),eyeData(:,8),eyeData(:,9),eyeData(:,10),'VariableNames',varNames);
% //     end
% //     TBsim = [TBsim;temp_TBsim];
% // end
% 
% // save('sim_ES_fixedALL','TBsim','paramset','final_subjlist')
% // close(h)
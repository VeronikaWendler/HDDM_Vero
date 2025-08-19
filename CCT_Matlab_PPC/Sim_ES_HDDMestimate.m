% % Clean up the environment
% %-----------------------------------------
clear all
clc
close all

m_num = 5
seed = sum(100*clock) + m_num + floor(1e6 * rand);
rng(seed);
%seed = rand; rng(seed);


% Load task-related information
%-----------------------------------------

data = readtable('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv');
               
data.OV = data.OV_2;
data.VD = data.VD_2;

% Convert 'cho' 1 = left, 2 = right into 'Choice' 0 = left, 1 = right
data.Choice = data.cho - 1;

% Rename 'corr' to 'Correct'
data.Correct = data.corr;

% **Remove rows with NaN values in the important columns**
cols_to_check = {'OV', 'VD', 'OV_2', 'VD_2','GazeDiff', 'FirstFixDur', 'FinalFixDur', 'MiddleFixDur', ...
                 'eachMiddleFixDur', 'GazeSwitch', 'FirstFixLoc', 'FinalFixLoc', ...
                 'DwellTimeAdvantage', 'chose_right'};

% Check that all specified columns exist in 'data' before using rmmissing
existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
cols_to_check = cols_to_check(existing_vars); % Keep only valid columns

% Apply rmmissing only to valid columns
data = rmmissing(data, 'DataVariables', cols_to_check);

% Filter only trials where phase is 'ES'
data = data(strcmp(data.phase, 'ES'), :);

% Define subject list (1 to 26 + 99), excluding specific subjects
subjlist = setdiff([1:26], [1, 4, 5, 6, 14, 99]);
nSubj = length(subjlist);
minSigma = randsample(0.02:0.001:0.03, nSubj, true); % HDDM assumes intra-trial variance is 1 for each time unit.


data.OV = data.OV_2;
data.VD = data.VD_2;

m_num = 5 % model5
    param_file = ['params_ES_m' num2str(m_num) '.csv'];
    converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
    conv_criteria = 1.1;

    %% read csv for parameters and converging document separately.
    %-----------------------------------------
    T = readtable(fullfile(param_file));
    Tconv = readtable(fullfile(converging_file));

    %% load individual parameter set
    %-----------------------------------------
    for k_subj = 1:nSubj
        clear paraname r
        subjID = subjlist(k_subj);
        paraname =  {['a_subj.' num2str(subjID)], ...
            ['t_subj(low).' num2str(subjID)], ...
            ['t_subj(medium).' num2str(subjID)], ...
            ['t_subj(high).' num2str(subjID)], ...
            ['v_Intercept_subj.' num2str(subjID)], ...
            ['v_AttentionW_subj.' num2str(subjID)], ...
            ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
            ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
            ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)]
            };

        r = find(ismember(T.Var1, paraname));
        a     = T.mean(r(1));
        ndt_H   = T.mean(r(2));
        ndt_L   = T.mean(r(3));
        ndt_M   = T.mean(r(4));
        beta0 = T.mean(r(5));
        beta1 = T.mean(r(6));
        beta2_H = T.mean(r(7));
        beta2_L = T.mean(r(8));
        beta2_M = T.mean(r(9));
        paramset(k_subj,:) = [a ndt_L ndt_M ndt_H beta0 beta1 beta2_L beta2_M beta2_H minSigma(k_subj)];
        
        %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
    end
    convergeset = Tconv.Gelman_Rubin;
    
    
    paraname_group_name =  {'a', ...
        't(low)', ...
        't(medium)', ...
        't(high)', ...
        'v_Intercept', ...
        'v_AttentionW', ...
        'v_InattentionW:C(OVcate)[low]', ...
        'v_InattentionW:C(OVcate)[medium]', ...
        'v_InattentionW:C(OVcate)[high]'
        };
    
    r_group = find(ismember(T.Var1, paraname_group_name));
    
    a     = T.mean(r_group(1));
    ndt_H   = T.mean(r_group(2));
    ndt_L   = T.mean(r_group(3));
    ndt_M   = T.mean(r_group(4));
    beta0 = T.mean(r_group(5));
    beta1 = T.mean(r_group(6));
    beta2_H = T.mean(r_group(7));
    beta2_L = T.mean(r_group(8));
    beta2_M = T.mean(r_group(9));
    theta(1,1) = beta2_L./beta1;
    theta(1,2) = beta2_M./beta1;
    theta(1,3) = beta2_H./beta1;
    
    paraname_group= [a ndt_L ndt_M ndt_H beta0 beta1 beta2_L beta2_M beta2_H theta];
%   
    
    


    save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m5_original','paramset','convergeset','paraname_group')


% Start simulation
load('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m5_original')
TBsim = [];
h = waitbar(0, 'Please wait...');
for k_subj = 1:nSubj
    count = 0;
    temp_TBsim = [];
    subjID = subjlist(k_subj);
    LMH    = [1 2 5 6 7; ...
              1 3 5 6 8; ...
              1 4 5 6 9];
%     LMH    = [1 2 5 6 7; ...
%         1 3 5 6 8; ...
%         1 4 5 6 9];
%     LMH = [ 1  2  3  6  9 ;   % k_OV = 1  (Low)
%         1  2  4  6  9 ;   % k_OV = 2  (Medium)
%         1  2  5  6  9 ];  % k_OV = 3  (High)
    waitbar(k_subj / nSubj)
    
    for k_OV = 1:3
        display(['subj' num2str(subjID) '_OV' num2str(k_OV)])
        params = squeeze(paramset(k_subj,LMH(k_OV,:)));
        
        data.OV = data.OV_2;     
        data.VD = data.VD_2;

        % Filter data for this subject & OV level
        data_subj = data(strcmp(data.phase, 'ES') & data.sub_id == subjID & data.OV == k_OV, {'phase', 'sub_id', 'VD', 'p1', 'p2', 'Correct', 'Choice'});
        values = [data_subj.p1, data_subj.p2]*1; % Check this ... Convert values to percentages
        
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
save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/sim_ES_m5_original','TBsim','paramset','subjlist')
close(h)

% 
% 





































% %% ES (model 5) simulation using empirical-like fixation sequences & durations
% % Integrates subject×OV fixation metrics into your ES accumulator.
% % Requires: fixation_metrics_ES.mat (from make_fixation_metrics_ES)
% 
% clear all; clc; close all;
% 
% m_num = 5;
% seed = sum(100*clock) + m_num + floor(1e6 * rand);
% rng(seed);
% 
% %% --- Paths (adjust if needed)
% data_file = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv';
% fix_file  = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/fixation_metrics_ES.mat';
% 
% param_file      = ['params_ES_m' num2str(m_num) '.csv'];
% converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
% 
% save_paramset_to = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m5_empFix';
% save_sim_to      = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/sim_ES_m5_empFix';
% 
% %% --- Load task data & clean (as in your original)
% data = readtable(data_file);
% 
% data.OV = data.OV_2;
% data.VD = data.VD_2;
% data.Choice  = data.cho - 1;
% data.Correct = data.corr;
% 
% cols_to_check = {'OV','VD','OV_2','VD_2','GazeDiff','FirstFixDur','FinalFixDur','MiddleFixDur', ...
%                  'eachMiddleFixDur','GazeSwitch','FirstFixLoc','FinalFixLoc', ...
%                  'DwellTimeAdvantage','chose_right'};
% existing_vars = ismember(cols_to_check, data.Properties.VariableNames);
% cols_to_check = cols_to_check(existing_vars);
% 
% data = rmmissing(data, 'DataVariables', cols_to_check);
% data = data(strcmp(data.phase,'ES'), :);
% 
% % subjects
% subjlist = setdiff(1:26, [1,4,5,6,14,99]);
% nSubj = numel(subjlist);
% minSigma = randsample(0.02:0.001:0.03, nSubj, true);
% 
% %% --- Load fixation metrics (built once via make_fixation_metrics_ES)
% load(fix_file, 'fixm');
% sub_ids   = fixm.sub_ids;
% ov_labels = fixm.ov_order;   % {'low','medium','high'} so 1->'low', 2->'medium', 3->'high'}
% 
% %% --- Load HDDM parameter sets for m5
% % m5 layout in your code:
% % paramset = [a ndt_L ndt_M ndt_H beta0 beta1 beta2_L beta2_M beta2_H minSigma]
% T    = readtable(fullfile(param_file));
% Tconv= readtable(fullfile(converging_file));
% 
% paramset = nan(nSubj, 10);
% for k_subj = 1:nSubj
%     subjID = subjlist(k_subj);
%     paraname =  {['a_subj.' num2str(subjID)], ...
%                  ['t_subj(low).' num2str(subjID)], ...
%                  ['t_subj(medium).' num2str(subjID)], ...
%                  ['t_subj(high).' num2str(subjID)], ...
%                  ['v_Intercept_subj.' num2str(subjID)], ...
%                  ['v_AttentionW_subj.' num2str(subjID)], ...
%                  ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
%                  ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
%                  ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)]};
%     r = find(ismember(T.Var1, paraname));
% 
%     a     = T.mean(r(1));
%     ndt_L = T.mean(r(2));
%     ndt_M = T.mean(r(3));
%     ndt_H = T.mean(r(4));
%     beta0 = T.mean(r(5));
%     beta1 = T.mean(r(6));
%     beta2_L = T.mean(r(7));
%     beta2_M = T.mean(r(8));
%     beta2_H = T.mean(r(9));
% 
%     paramset(k_subj,:) = [a ndt_L ndt_M ndt_H beta0 beta1 beta2_L beta2_M beta2_H minSigma(k_subj)];
% end
% 
% convergeset = Tconv.Gelman_Rubin;
% 
% % group-level (for completeness; you used it before)
% paraname_group_name =  {'a','t(low)','t(medium)','t(high)','v_Intercept','v_AttentionW', ...
%                         'v_InattentionW:C(OVcate)[low]','v_InattentionW:C(OVcate)[medium]','v_InattentionW:C(OVcate)[high]'};
% r_group = find(ismember(T.Var1, paraname_group_name));
% a     = T.mean(r_group(1));
% ndt_L = T.mean(r_group(2));
% ndt_M = T.mean(r_group(3));
% ndt_H = T.mean(r_group(4));
% beta0 = T.mean(r_group(5));
% beta1 = T.mean(r_group(6));
% beta2_L = T.mean(r_group(7));
% beta2_M = T.mean(r_group(8));
% beta2_H = T.mean(r_group(9));
% theta = [beta2_L./beta1, beta2_M./beta1, beta2_H./beta1];
% paraname_group = [a ndt_L ndt_M ndt_H beta0 beta1 beta2_L beta2_M beta2_H theta];
% 
% save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Sim/with_accumulation/Sim_HDDM_ES_m5_empFix_paramset','paramset','convergeset','paraname_group');
% 
% %% --- Simulation (per subject × OV × trial)
% TBsim = [];
% h = waitbar(0, 'Simulating ES (m5) with empirical fixations...');
% 
% % param indices per OV (your original LMH)
% % OV=1 -> [a ndt_L beta0 beta1 beta2_L]
% % OV=2 -> [a ndt_M beta0 beta1 beta2_M]
% % OV=3 -> [a ndt_H beta0 beta1 beta2_H]
% LMH = [1 2 5 6 7; ...
%        1 3 5 6 8; ...
%        1 4 5 6 9];
% 
% for k_subj = 1:nSubj
%     waitbar(k_subj/nSubj, h);
%     subjID = subjlist(k_subj);
% 
%     for k_OV = 1:3
%         fprintf('subj %d  |  OV %d\n', subjID, k_OV);
% 
%         % select the right subset of params for this OV
%         params_all = squeeze(paramset(k_subj,:));
%         params = params_all(LMH(k_OV,:));   % [a, ndt(ov), beta0, beta1, beta2(ov)]
% 
%         % pull fixation metrics (subject×OV with group fallback)
%         ov_label = ov_labels{k_OV};
%         si = find(sub_ids == subjID, 1, 'first');
%         oi = find(strcmp(ov_labels, ov_label), 1, 'first');
%         if isempty(oi), error('OV label not found in fixm.ov_order'); end
% 
%         if isempty(si)
%             warning('Subject %d not in fixation metrics; using group fallback.', subjID);
%             S = fixm.groupOV(oi); G = fixm.groupOV(oi);
%         else
%             S = fixm.by_subjOV(si, oi); G = fixm.groupOV(oi);
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
%         % subject × OV trials
%         data_subj = data(data.sub_id==subjID & data.OV==k_OV, {'phase','sub_id','VD','p1','p2','Correct','Choice'});
%         values    = [data_subj.p1, data_subj.p2] *1;  % your original had *100 for ES m5
% 
%         count = 0; clear behData eyeData
%         for ktrial = 1:height(data_subj)
%             Vl = values(ktrial,1);
%             Vr = values(ktrial,2);
% 
%             % sample Nfix from pmf (or use empirical Nfix if you have it)
%             Nfix = sample_from_pmf(nfix_vals_s, nfix_pmf_s);
% 
%             % gaze side sequence + durations
%             FixR_seq   = simulate_fixation_sequence(Vl, Vr, Nfix, ...
%                             p_firstR_VrGt, p_firstR_VlGt, p_firstR_Eq, p_switch);
% 
%             FixDur_seq = sample_fixation_durations_LN(Nfix, ...
%                             ln_mu_first, ln_sg_first, ln_mu_middle, ln_sg_middle, ln_mu_final, ln_sg_final);
% 
%             % accumulate (NOTE: EvidenceAccumulate_ES must accept the 2 extra args)
%             [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = ...
%                 EvidenceAccumulate_ES2(k_OV, Vl, Vr, params, FixR_seq, FixDur_seq);
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
% save([save_sim_to '_empiricalFix'], 'TBsim','paramset','subjlist');
% 
% %% ----------------- Helper functions (local; defined before use) -----------------
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
%     ps = max(min(p_switch, 0.999), 0.001);
%     for t = 2:Nfix
%         if rand < ps
%             FixR_seq(t) = 1 - FixR_seq(t-1); % switch
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
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 





































% %     %-----------------------------------------
% %     for k_subj = 1:nSubj
% %         clear paraname r
% %         subjID = subjlist(k_subj);
% %         paraname =  {['a_subj(low).' num2str(subjID)], ...
% %             ['a_subj(medium).' num2str(subjID)], ...
% %             ['a_subj(high).' num2str(subjID)], ...
% %             ['t_subj.' num2str(subjID)], ...
% %             ['v_Intercept_subj.' num2str(subjID)], ...
% %             ['v_AttentionW_subj.' num2str(subjID)], ...
% %             ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
% %             ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
% %             ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)]
% %             };
% % 
% %         r = find(ismember(T.Var1, paraname));
% %         a_H     = T.mean(r(1));
% %         a_L   = T.mean(r(2));
% %         a_M   = T.mean(r(3));
% %         ndt   = T.mean(r(4));
% %         beta0 = T.mean(r(5));
% %         beta1 = T.mean(r(6));
% %         beta2_H = T.mean(r(7));
% %         beta2_L = T.mean(r(8));
% %         beta2_M = T.mean(r(9));
% %         paramset(k_subj,:) = [a_L a_M a_H ndt beta0 beta1 beta2_L beta2_M beta2_H minSigma(k_subj)];
% %         
% %         %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
% %     end
% %     convergeset = Tconv.Gelman_Rubin;
% %     
% %     
% %     paraname_group_name =  {'a(low)', ...
% %         'a(medium)', ...
% %         'a(high)', ...
% %         't', ...
% %         'v_Intercept', ...
% %         'v_AttentionW', ...
% %         'v_InattentionW:C(OVcate)[low]', ...
% %         'v_InattentionW:C(OVcate)[medium]', ...
% %         'v_InattentionW:C(OVcate)[high]'
% %         };
% %     
% %     r_group = find(ismember(T.Var1, paraname_group_name));
% %     
% %     a_H     = T.mean(r_group(1));
% %     a_L   = T.mean(r_group(2));
% %     a_M   = T.mean(r_group(3));
% %     ndt   = T.mean(r_group(4));
% %     beta0 = T.mean(r_group(5));
% %     beta1 = T.mean(r_group(6));
% %     beta2_H = T.mean(r_group(7));
% %     beta2_L = T.mean(r_group(8));
% %     beta2_M = T.mean(r_group(9));
% %     theta(1,1) = beta2_L./beta1;
% %     theta(1,2) = beta2_M./beta1;
% %     theta(1,3) = beta2_H./beta1;
% %     
% %     paraname_group= [a_L a_M a_H ndt beta0 beta1 beta2_L beta2_M beta2_H theta];
% % 
% % %     for k_subj = 1:nSubj
% % %         clear paraname r
% % %         subjID = subjlist(k_subj);
% % %         paraname =  {['a_subj.' num2str(subjID)], ...
% % %             ['t_subj.' num2str(subjID)], ...
% % %             ['v_Intercept_subj.' num2str(subjID)], ...
% % %             ['v_AttentionW_subj.' num2str(subjID)], ...
% % %             ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
% % %             ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
% % %             ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)]
% % %             };
% % % 
% % %         r = find(ismember(T.Var1, paraname));
% % %         a     = T.mean(r(1));
% % %         ndt   = T.mean(r(2));
% % %         beta0 = T.mean(r(3));
% % %         beta1 = T.mean(r(4));
% % %         beta2_H = T.mean(r(5));
% % %         beta2_L = T.mean(r(6));
% % %         beta2_M = T.mean(r(7));
% % %         paramset(k_subj,:) = [a ndt beta0 beta1 beta2_L beta2_M beta2_H minSigma(k_subj)];
% % %         
% % %         %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
% % %     end
% % %     convergeset = Tconv.Gelman_Rubin;
% % %     
% % %     
% % %     paraname_group_name =  {'a', ...
% % %         't', ...
% % %         'v_Intercept', ...
% % %         'v_AttentionW', ...
% % %         'v_InattentionW:C(OVcate)[low]', ...
% % %         'v_InattentionW:C(OVcate)[medium]', ...
% % %         'v_InattentionW:C(OVcate)[high]'
% % %         };
% % %     
% % %     r_group = find(ismember(T.Var1, paraname_group_name));
% % %     
% % %     a     = T.mean(r_group(1));
% % %     ndt   = T.mean(r_group(2));
% % %     beta0 = T.mean(r_group(3));
% % %     beta1 = T.mean(r_group(4));
% % %     beta2_H = T.mean(r_group(5));
% % %     beta2_L = T.mean(r_group(6));
% % %     beta2_M = T.mean(r_group(7));
% % %     theta(1,1) = beta2_L./beta1;
% % %     theta(1,2) = beta2_M./beta1;
% % %     theta(1,3) = beta2_H./beta1;
% % %     
% % %     paraname_group= [a ndt beta0 beta1 beta2_L beta2_M beta2_H theta];
% 
% 
% m_num = 14 % model5
%     param_file = ['params_ES_m' num2str(m_num) '.csv'];
%     converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
%     conv_criteria = 1.01;
% 
%     %% read csv for parameters and converging document separately.
%     %-----------------------------------------
%     T = readtable(fullfile(param_file));
%     Tconv = readtable(fullfile(converging_file));
% 
%     %% load individual parameter set
%     %-----------------------------------------
% 
%     for k_subj = 1:nSubj
%         clear paraname r
%         subjID = subjlist(k_subj);
%         paraname =  {['a_subj.' num2str(subjID)], ...
%             ['t_subj.' num2str(subjID)], ...
%             ['v_Intercept_subj.' num2str(subjID)], ...
%             ['v_AttentionW_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)], ...
%             ['z_subj(0).' num2str(subjID)], ...  
%             ['z_subj(1).' num2str(subjID)]      
%             };
%         [found,loc] = ismember(paraname, T.Var1);
%         if ~all(found)
%             warning('Missing parameters:\n   %s\n', strjoin(paraname(~found), '\n   '))
%         end
% 
% 
%         r = find(ismember(T.Var1, paraname));
%         a     = T.mean(r(1));
%         ndt   = T.mean(r(2));
%         beta0 = T.mean(r(3));
%         beta1 = T.mean(r(4));
%         beta2_H = T.mean(r(5));
%         beta2_L = T.mean(r(6));
%         beta2_M = T.mean(r(7));
%         z0 = T.mean(r(8));
%         z1 = T.mean(r(9));
%         paramset(k_subj,:) = [a ndt beta0 beta1 beta2_L beta2_M beta2_H z0 z1 minSigma(k_subj)];
%         
%         %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
%     end
%     convergeset = Tconv.Gelman_Rubin;
%     
%     
%     paraname_group_name =  {'a', ...
%         't', ...
%         'v_Intercept', ...
%         'v_AttentionW', ...
%         'v_InattentionW:C(OVcate)[low]', ...
%         'v_InattentionW:C(OVcate)[medium]', ...
%         'v_InattentionW:C(OVcate)[high]', ...
%         'z(0)', ...  
%         'z(1)'
%         };
%     
%     r_group = find(ismember(T.Var1, paraname_group_name));
%     
%     a     = T.mean(r_group(1));
%     ndt   = T.mean(r_group(2));
%     beta0 = T.mean(r_group(3));
%     beta1 = T.mean(r_group(4));
%     beta2_H = T.mean(r_group(5));
%     beta2_L = T.mean(r_group(6));
%     beta2_M = T.mean(r_group(7));
%     z0 = T.mean(r_group(8));
%     z1 = T.mean(r_group(9));
%     theta(1,1) = beta2_L./beta1;
%     theta(1,2) = beta2_M./beta1;
%     theta(1,3) = beta2_H./beta1;
% 
%     
%     paraname_group= [a ndt beta0 beta1 beta2_L beta2_M beta2_H z0 z1 theta];
% %   
% m_num = 24 % model5
%     param_file = ['params_ES_m' num2str(m_num) '.csv'];
%     converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
%     conv_criteria = 1.01;
% 
%     %-----------------------------------------
%     T = readtable(fullfile(param_file));
%     Tconv = readtable(fullfile(converging_file));
% 
%     %-----------------------------------------
% 
%     for k_subj = 1:nSubj
%         clear paraname r
%         subjID = subjlist(k_subj);
%         paraname =  {
%             ['v_Intercept_subj.' num2str(subjID)], ...
%             ['v_AttentionW_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[low]_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[medium]_subj.' num2str(subjID)], ...
%             ['v_InattentionW:C(OVcate)[high]_subj.' num2str(subjID)], ...
%             ['a_Intercept_subj.' num2str(subjID)], ...
%             ['a_C(OVcate)[T.low]_subj.' num2str(subjID)], ...
%             ['a_C(OVcate)[T.medium]_subj.' num2str(subjID)], ...
%             ['t_Intercept_subj.' num2str(subjID)], ...
%             ['t_C(OVcate)[T.low]_subj.' num2str(subjID)], ...
%             ['t_C(OVcate)[T.medium]_subj.' num2str(subjID)], ...
%             ['z_subj(0).' num2str(subjID)], ...  
%             ['z_subj(1).' num2str(subjID)]      
%             };
%         [found,loc] = ismember(paraname, T.Var1);
%         if ~all(found)
%             warning('Missing parameters:\n   %s\n', strjoin(paraname(~found), '\n   '))
%         end
% 
% 
%         r = find(ismember(T.Var1, paraname));
%         vbeta0     = T.mean(r(1));
%         vbeta1     = T.mean(r(2));
%         vbeta2_H    = T.mean(r(3));
%         vbeta2_L    = T.mean(r(4));
%         vbeta2_M    = T.mean(r(5));
%         abeta0      = T.mean(r(6));
%         abeta2_L    = T.mean(r(7));
%         abeta2_M    = T.mean(r(8));
%         tbeta0      = T.mean(r(9));
%         tbeta2_L    = T.mean(r(10));
%         tbeta2_M    = T.mean(r(11));
%         z0          = T.mean(r(12));
%         z1          = T.mean(r(13));
%         paramset(k_subj,:) = [vbeta0 vbeta1 vbeta2_L vbeta2_M vbeta2_H abeta0 abeta2_L abeta2_M tbeta0 tbeta2_L tbeta2_M z0 z1 minSigma(k_subj)];
%         
%         %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
%     end
%     convergeset = Tconv.Gelman_Rubin;  
%     
%     paraname_group_name =  {
%         'v_Intercept', ...
%         'v_AttentionW', ...
%         'v_InattentionW:C(OVcate)[low]', ...
%         'v_InattentionW:C(OVcate)[medium]', ...
%         'v_InattentionW:C(OVcate)[high]', ...
%         'a_Intercept', ...
%         'a_C(OVcate)[T.low]', ...
%         'a_C(OVcate)[T.medium]', ...
%         't_Intercept', ...
%         't_C(OVcate)[T.low]', ...
%         't_C(OVcate)[T.medium]', ...
%         'z(0)', ...  
%         'z(1)'
%         };
%     
%     r_group = find(ismember(T.Var1, paraname_group_name));
%     
% 
%     vbeta0     = T.mean(r_group(1));
%     vbeta1     = T.mean(r_group(2));
%     vbeta2_H    = T.mean(r_group(3));
%     vbeta2_L    = T.mean(r_group(4));
%     vbeta2_M    = T.mean(r_group(5));
%     abeta0      = T.mean(r_group(6));
%     abeta2_L    = T.mean(r_group(7));
%     abeta2_M    = T.mean(r_group(8));
%     tbeta0      = T.mean(r_group(9));
%     tbeta2_L    = T.mean(r_group(10));
%     tbeta2_M    = T.mean(r_group(11));
%     z0          = T.mean(r_group(12));
%     z1          = T.mean(r_group(13));
%     theta(1,1) = vbeta2_L./vbeta1;
%     theta(1,2) = vbeta2_M./vbeta1;
%     theta(1,3) = vbeta2_H./vbeta1;
% 
%     
%     paraname_group= [vbeta0 vbeta1 vbeta2_L vbeta2_M vbeta2_H abeta0 abeta2_L abeta2_M tbeta0 tbeta2_L tbeta2_M z0 z1 theta];
%   
%   

% m_num = 5 % model5
%     param_file = ['params_ES_m' num2str(m_num) '.csv'];
%     converging_file = ['gelman_rubin_ES_m' num2str(m_num) '.csv'];
%     conv_criteria = 1.01;
% 
%     %% read csv for parameters and converging document separately.
%     %-----------------------------------------
%     T = readtable(fullfile(param_file));
%     Tconv = readtable(fullfile(converging_file));
% 
%     %% load individual parameter set
%     %-----------------------------------------
% 
%     for k_subj = 1:nSubj
%         clear paraname r
%         subjID = subjlist(k_subj);
%         paraname =  {['t_subj.' num2str(subjID)], ...
%             ['v_Intercept_subj.' num2str(subjID)], ...
%             ['v_val_diff_subj.' num2str(subjID)], ...
%             ['v_DwellPropAdvantage_subj.' num2str(subjID)], ...
%             ['v_gaze_quad_subj.' num2str(subjID)], ...
%             ['a_Intercept_subj.' num2str(subjID)], ...
%             ['a_abs_DwellPropAdv:C(OVcate)[high]_subj.' num2str(subjID)], ...
%             ['a_abs_DwellPropAdv:C(OVcate)[low]_subj.' num2str(subjID)], ...
%             ['a_abs_DwellPropAdv:C(OVcate)[medium]_subj.' num2str(subjID)]
%             };
% 
%         [found,loc] = ismember(paraname, T.Var1);
%         if ~all(found)
%             warning('Missing parameters:\n   %s\n', strjoin(paraname(~found), '\n   '))
%         end
% 
%         r = find(ismember(T.Var1, paraname));
%         ndt        = T.mean(r(1));
%         vbeta0     = T.mean(r(2));
%         vbeta1     = T.mean(r(3));
%         vbeta2     = T.mean(r(4));
%         vbeta3     = T.mean(r(5));
%         abeta0     = T.mean(r(6));
%         abeta1_H   = T.mean(r(7));
%         abeta1_L   = T.mean(r(8));
%         abeta1_M   = T.mean(r(9));
%         paramset(k_subj,:) = [ndt vbeta0 vbeta1 vbeta2 vbeta3 abeta0 abeta1_H abeta1_L abeta1_M minSigma(k_subj)];
%         
%         %         convergetest(k_OV,k_subj,:) = convergeset(k_OV,k_subj,:)>=conv_criteria;
%     end
%     convergeset = Tconv.Gelman_Rubin;
%     
%     
%     paraname_group_name =  {
%         't', ...
%         'v_Intercept', ...
%         'v_val_diff', ...
%         'v_DwellPropAdvantage', ...
%         'v_gaze_quad', ...
%         'a_Intercept', ...
%         'a_abs_DwellPropAdv:C(OVcate)[high]', ...
%         'a_abs_DwellPropAdv:C(OVcate)[low]', ...  
%         'a_abs_DwellPropAdv:C(OVcate)[medium]'
%         };
%     
%     r_group = find(ismember(T.Var1, paraname_group_name));
%     
%     ndt        = T.mean(r_group(1));
%     vbeta0     = T.mean(r_group(2));
%     vbeta1     = T.mean(r_group(3));
%     vbeta2     = T.mean(r_group(4));
%     vbeta3     = T.mean(r_group(5));
%     abeta0     = T.mean(r_group(6));
%     abeta1_H   = T.mean(r_group(7));
%     abeta1_L   = T.mean(r_group(8));
%     abeta1_M   = T.mean(r_group(9));
%     
%     paraname_group= [ndt vbeta0 vbeta1 vbeta2 vbeta3 abeta0 abeta1_H abeta1_L abeta1_M];



% This code computes mean and STD per subject and per OV level,
% then lognormal transforms them (Ting & Gluth, 2024 style)

% input/output paths
input_file  = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv';
output_file = 'D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Fixation_mean_log_simulation_ES_ZBIAS.csv';

% dataset
data = readtable(input_file);

% parameters
exclude_subjects = [1, 4, 5, 6, 14, 99];    % Subjects to exclude
phase_filter    = 'ES';                      % Experimental phase filter
ov_categories   = {'low', 'medium', 'high'};  % OV levels

data = data(strcmp(data.phase, phase_filter), :);
data(ismember(data.sub_id, exclude_subjects), :) = [];
if iscell(data.rtime)
    data.rtime = cellfun(@str2double, data.rtime);
end
data = data(data.rtime >= 0.250, :);

%% Convert columns to numeric in case theyre not 
numeric_cols = {'p1', 'p2', 'FirstFixLoc', 'FirstFixDur', 'FinalFixDur', 'eachMiddleFixDur'};
for i = 1:numel(numeric_cols)
    col = data.(numeric_cols{i});
    if isnumeric(col)
        continue;
    elseif iscell(col)
        first_elem = col{1};
        test_val   = str2double(first_elem);
        if isnan(test_val)
            newcol = cellfun(@(x) mean(str2num(x)), col, 'UniformOutput', true);
        else
            newcol = cellfun(@(x) str2double(x), col, 'UniformOutput', true);
        end
        data.(numeric_cols{i}) = newcol;
    else
        data.(numeric_cols{i}) = str2double(col);
    end
end

ov_order = {'low','medium','high'};
data.OVcate_2 = categorical(data.OVcate_2, ov_order, 'Ordinal', true);

% Remove nans
data = rmmissing(data, 'DataVariables', [numeric_cols, {'OVcate_2'}]);

%% OV first-fixation proportions across subjects
[G_sub, subjects] = findgroups(data.sub_id);
p_rights = splitapply(@(x) mean(x==2,'omitnan'), data.FirstFixLoc, G_sub);
p_lefts  = splitapply(@(x) mean(x==1,'omitnan'), data.FirstFixLoc, G_sub);

nSubj        = numel(subjects);
mean_p_right = mean(p_rights);
std_p_right  = std(p_rights);
mean_p_left  = mean(p_lefts);
std_p_left   = std(p_lefts);

fprintf('Overall (n=%d subjects):\n', nSubj);
fprintf('  Mean %% first-fix right = %.2f%% (SD = %.2f%%)\n', mean_p_right*100, std_p_right*100);
fprintf('  Mean %% first-fix left  = %.2f%% (SD = %.2f%%)\n', mean_p_left*100,  std_p_left*100);

% results table for OV-level metrics
results = table('Size',[numel(ov_categories),14], ...
    'VariableTypes',{'string','double','double','double','double', ...
                    'double','double','double','double', ...
                    'double','double','double','double','double'}, ...
    'VariableNames',{'OVcate_2', ...
                    'P_FixRight_VrGreater','SD_FixRight_VrGreater', ...
                    'P_FixRight_VlGreater','SD_FixRight_VlGreater', ...
                    'Mean_FirstFixDur','SD_FirstFixDur','Mu_FirstFixDur','Sigma_FirstFixDur', ...
                    'Mean_RemainingDur','SD_RemainingDur','Mu_RemainingDur','Sigma_RemainingDur', ...
                    'N_subj'});
results.OVcate_2 = string(ov_categories(:));

%% Subject-level, per-OV metrics grouping
[G_joint, subjIDs, ovLevels] = findgroups(data.sub_id, data.OVcate_2);
subj_ov_firstDur = splitapply(@(x) mean(x,'omitnan'), data.FirstFixDur, G_joint);
remDurs           = data.FinalFixDur + data.eachMiddleFixDur;
subj_ov_remDur    = splitapply(@(x) mean(x,'omitnan'), remDurs, G_joint);

isVr = data.p2 > data.p1;
isVl = data.p2 < data.p1;
isR  = data.FirstFixLoc == 2;
pr_vr = splitapply(@(vg,ff) mean(ff(vg),'omitnan'), isVr, isR, G_joint);
pr_vl = splitapply(@(vl,ff) mean(ff(vl),'omitnan'), isVl, isR, G_joint);

for j = 1:numel(ov_categories)
    ov         = ov_categories{j};
    sel        = (ovLevels == ov);
    subj_count = sum(sel);
    results.N_subj(j) = subj_count;
    if subj_count == 0
        continue;
    end

    % Probabilities
    vr_subj = pr_vr(sel);
    vl_subj = pr_vl(sel);
    tot     = vr_subj + vl_subj;
    valid   = tot > 0;
    vr_subj(valid) = vr_subj(valid) ./ tot(valid);
    vl_subj(valid) = vl_subj(valid) ./ tot(valid);

    results.P_FixRight_VrGreater(j) = mean(vr_subj,'omitnan');
    results.SD_FixRight_VrGreater(j) = std(vr_subj,'omitnan');
    results.P_FixRight_VlGreater(j) = mean(vl_subj,'omitnan');
    results.SD_FixRight_VlGreater(j) = std(vl_subj,'omitnan');

    %First-fix dur
    ffd = subj_ov_firstDur(sel);
    m1  = mean(ffd,'omitnan');  s1 = std(ffd,'omitnan');
    results.Mean_FirstFixDur(j)  = m1;
    results.SD_FirstFixDur(j)    = s1;
    results.Mu_FirstFixDur(j)    = log(m1);
    results.Sigma_FirstFixDur(j) = sqrt(log(1 + (s1/m1)^2));

    % Rest of dur
    rmd = subj_ov_remDur(sel);
    m2  = mean(rmd,'omitnan');  s2 = std(rmd,'omitnan');
    results.Mean_RemainingDur(j)  = m2;
    results.SD_RemainingDur(j)    = s2;
    results.Mu_RemainingDur(j)    = log(m2);
    results.Sigma_RemainingDur(j) = sqrt(log(1 + (s2/m2)^2));
end

%% csv
writetable(results, output_file);
disp('Fixation Statistics by OV Level:');
disp(results);


subj_ids = unique(data.sub_id);
nSubj    = numel(subj_ids);
p_right  = NaN(nSubj,1);
p_left   = NaN(nSubj,1);

for i = 1:nSubj
    s = subj_ids(i);
    d = data(data.sub_id==s, :);
    % proportion of trials for this subject where FirstFixLoc == 2
    p_right(i) = mean(d.FirstFixLoc == 2, 'omitnan');
    p_left(i)  = mean(d.FirstFixLoc == 1, 'omitnan');
end

% group-level summary
mean_p_right = mean(p_right);
std_p_right  = std(p_right);
mean_p_left  = mean(p_left);
std_p_left   = std(p_left);
fprintf('Across %d subjects:\n', nSubj);
fprintf('  Mean %% first‐fix right  = %.2f%%  (SD = %.2f%%)\n', mean_p_right*100, std_p_right*100);
fprintf('  Mean %% first‐fix left   = %.2f%%  (SD = %.2f%%)\n', mean_p_left*100,  std_p_left*100);
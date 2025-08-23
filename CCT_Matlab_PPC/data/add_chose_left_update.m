%This program adds an extra column 'trial_type' for categorising early and late trials in teh ES phase

file = 'GarciaData_eye_organization_2.mat';

S = load(file);
TB = S.TB;
G = TB.Garcia;

% Filter for phase and exclude subjects
excluded_subjects = [1, 4, 5, 6, 14, 99];
phase_filter = "ES";

idx_keep = strcmp(G.phase, phase_filter) & ...
           ~ismember(G.sub_id, excluded_subjects) & ...
           ~isnan(G.rtime);  % Only care about rtime

G_valid = G(idx_keep, :); % filtered
ET_LT = strings(height(G_valid), 1);  % 'ET' or 'LT'
sub_ids = unique(G_valid.sub_id);

for i = 1:length(sub_ids)
    sid = sub_ids(i);
    idx_sub = G_valid.sub_id == sid;
    rt_sub = G_valid.rtime(idx_sub);
    rt_median = median(rt_sub, 'omitnan');

    % Get global row indices for subject's trials
    global_idx = find(idx_sub);

    % Assign "ET" by default
    ET_LT(global_idx) = "ET";

    % Overwrite with "LT" for trials above median
    lt_idx_within = rt_sub > rt_median;
    ET_LT(global_idx(lt_idx_within)) = "LT";
end

G_valid.trial_type = ET_LT;
G.trial_type = repmat("", height(G), 1); % Initialize empty
G.trial_type(idx_keep) = G_valid.trial_type;

% Save
TB.Garcia = G;
S.TB = TB;
[folder, name, ext] = fileparts(file);
if isempty(folder), folder = pwd; end
backup = fullfile(folder, sprintf('%s_backup_%s%s', name, datestr(now, 'yyyymmdd_HHMMSS'), ext));
copyfile(file, backup);
save(file, '-struct', 'S', '-v7.3');

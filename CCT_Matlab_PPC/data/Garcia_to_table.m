% transforms the csv to mat

load('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/data/Garcia_Eye_for_Simulation.mat');

% data = TB.Garcia;
% 
% % Get field names
% fields = fieldnames(data);
% 
% % Initialize an empty table
% GarciaTable = table();
% 
% for i = 1:length(fields)
%     fieldName = fields{i};
%     fieldData = data.(fieldName);
%     
%     if isvector(fieldData) && size(fieldData, 1) == 1
%         fieldData = fieldData'; % Transpose to column vector
%     end
%     
%     if iscell(fieldData)
%         fieldData = string(fieldData);
%     end
%     
%     GarciaTable.(fieldName) = fieldData;
% end
% 
% % Save table format
% TB.Garcia = GarciaTable;
% 
%  Save back to a .mat file
% save('C:/Cluster_Github/HDDM_Vero/CCT_Matlab_PPC/data/GarciaData_Transformed.mat', 'TB');





% Extract the struct
%data = TB.Garcia;

%fields = fieldnames(data);

%GarciaTable = table();

for i = 1:length(fields)
%    fieldName = fields{i};
%    fieldData = data.(fieldName);

%    % field is row vector before transposing
%    if isvector(fieldData) && size(fieldData, 1) == 1
%        fieldData = fieldData'; % Transpose to column vector
%    end
%    if iscell(fieldData)
%        fieldData = string(fieldData);
%    end
%    if (strcmp(fieldName, 'p1') || strcmp(fieldName, 'p2')) && isnumeric(fieldData)
%        fieldData = fieldData / 100;  % Convert from e.g. 70 -> 0.70
%    end

%    GarciaTable.(fieldName) = fieldData;
%end
%TB.Garcia = GarciaTable;
%save('D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/stats_TingGluth/Analysis_Simulation_replication/simulation/Data/GarciaData_Transformed.mat', 'TB');
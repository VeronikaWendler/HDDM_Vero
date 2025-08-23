function plotEmpiricalData(matOrCsvFile)
% plotEmpiricalData  Load CSV or MAT, plot, save 
% CSV:  sub_id, phase, rtime, cho(1=E,2=S), DwellTimeAdvantage (this is S-E )
% SIM:  sub_id, (no phase), RT, Choice(0=E,1=S), DwellDiff (S-E same as R-L)

  % locate / choose file
  if nargin<1 || isempty(matOrCsvFile)
    dataDir = fullfile(pwd,'data_BGarcia');
    files   = [dir(fullfile(dataDir,'*.csv')); dir(fullfile(dataDir,'*.mat'))];
    if isempty(files), error('No CSV or MAT found in %s',dataDir); end
    if numel(files)>1
      [f,p] = uigetfile({'*.csv;*.mat','Data files (*.csv,*.mat)'}, dataDir);
      if f==0, return; end
      matOrCsvFile = fullfile(p,f);
    else
      matOrCsvFile = fullfile(dataDir,files(1).name);
    end
  end

  % read table
  [~,~,ext] = fileparts(matOrCsvFile);
  switch lower(ext)
    case '.csv'
      T = readtable(matOrCsvFile);
    case '.mat'
      T = loadTableFromMat(matOrCsvFile);
    otherwise
      error('Unsupported extension %s',ext);
  end

  % figure out format by variable names
  vars = T.Properties.VariableNames;
  isReal = ismember('cho',vars);      % in my empirical (real) CSV data this is the name
  isSim  = ismember('Choice',vars);   % simulated MAT (similar to Chih-Chung) 


  % filter subjects and ES phase 
  exclude = [1,4,5,6,14,99];  
  maskExclude = ~ismember(T.sub_id,exclude);

  if ismember('phase',vars)
      maskPhase = strcmp(T.phase,'ES');
  else
      maskPhase = true(size(maskExclude)); % sim files already ES-only
  end

% if interested in specific fixation constraints 
%   maskFix = true(height(T),1);
%   if strcmpi(ext,'.csv')
%       needCols = {'RightFixNR','LeftFixNR'};
%       if all(ismember(needCols, vars))
%           maskFix = (T.RightFixNR > 0) & (T.LeftFixNR > 0);
%       else
%           warning('skipping fix-count filter');
%       end
%   end
% 
%   T = T(maskExclude & maskPhase & maskFix, :);

  T = T(maskExclude & maskPhase,:);

  % harmonize columns
  if isReal
      RTvec      = T.rtime;
      choiceEall = (T.cho==1);                  % 1 = E chosen (in my empirical csv data, cho is 2 for S and 1 for E chosen)
      dwaSall    = T.DwellTimeAdvantage;        % S - E
  else % sim
      RTvec      = T.rt;
      choiceEall = (T.Choice==0);               % 0 = E chosen
      dwaSall    = T.DwellDiff;                 % S - E
  end
 
  %RT restriction criteria
  rtMask = (RTvec >= 0) & (RTvec <= 8);

  RTvec      = RTvec(rtMask);
  choiceEall = choiceEall(rtMask);
  dwaSall    = dwaSall(rtMask);
  T          = T(rtMask,:);

  % analysis - unchanged (this is similar to Sebastian's code)
  subs     = unique(T.sub_id);
  nS       = numel(subs);
  pE_byS   = nan(nS,5);
  pS_byS   = nan(nS,5);

  for i = 1:nS
    sid  = subs(i);
    sel  = T.sub_id==sid;
    RTs  = RTvec(sel);
    chE  = choiceEall(sel);
    edges = quantile(RTs,4);
    rb    = discretize(RTs,[-inf edges inf]);
    for b=1:5
      pE_byS(i,b) = mean(   chE(rb==b) );
      pS_byS(i,b) = mean( ~ chE(rb==b) );
    end
  end

  m_qRT_E   = mean(pE_byS,1);    sem_qRT_E = std(pE_byS,0,1)/sqrt(nS);
  m_qRT_S   = mean(pS_byS,1);    sem_qRT_S = std(pS_byS,0,1)/sqrt(nS);

  edgesD = quantile(dwaSall,4);
  db     = discretize(dwaSall,[-inf edgesD inf]);
  m_pS   = nan(1,5); sem_pS = nan(1,5);
  m_RT   = nan(1,5); sem_RT = nan(1,5);
  m_RT_E = nan(1,5); sem_RT_E = nan(1,5);
  m_RT_S = nan(1,5); sem_RT_S = nan(1,5);

  for b=1:5
    idx        = (db==b);
    N          = sum(idx);
    pS         = mean(~choiceEall(idx));
    m_pS(b)    = pS;
    sem_pS(b)  = sqrt(pS*(1-pS)/N);

    theseRT    = RTvec(idx);
    m_RT(b)    = mean(theseRT);
    sem_RT(b)  = std(theseRT)/sqrt(N);

    idxE       = idx &  choiceEall;
    idxS       = idx & ~choiceEall;

    RT_E       = RTvec(idxE);
    m_RT_E(b)  = mean(RT_E);
    sem_RT_E(b)= std(RT_E)/sqrt(numel(RT_E));

    RT_S       = RTvec(idxS);
    m_RT_S(b)  = mean(RT_S);
    sem_RT_S(b)= std(RT_S)/sqrt(numel(RT_S));
  end

  % rename for plotting
  m_qRT   = m_qRT_E;
  sem_qRT = sem_qRT_E;

  % Accuracy vs Dwell Time Advantage for Correct Option (assumes Vl, Vr) --
  m_corrProb  = NaN(1,5);
  sem_corrProb= NaN(1,5);

  if isSim && ismember('DwellDiff', T.Properties.VariableNames) && ismember('Correct', T.Properties.VariableNames)
    
      if ~all(ismember({'Vl','Vr'}, T.Properties.VariableNames))
          warning('Skipping plot %s', strjoin(T.Properties.VariableNames, ', '));
      else
     
          valid = ~isnan(T.DwellDiff) & (T.Vl ~= T.Vr);

          % dwell-time advantage for the correct option
          DAC = sign(T.Vr(valid) - T.Vl(valid)) .* T.DwellDiff(valid);  % >0  dwell favors higher-value option
          ACC = T.Correct(valid);                                       % 1 = chose higher value

          % need enough data
          if numel(DAC) >= 5
              % 5 bins quantiles
              q = prctile(DAC,[20 40 60 80]);
              epsStep = max(1, range(DAC))*1e-12;
              for k = 2:numel(q)
                  if q(k) <= q(k-1), q(k) = q(k-1) + epsStep; end
              end
              edges = [-inf q inf];
              bins  = discretize(DAC, edges);

              for b = 1:5
                  idx = (bins == b);
                  N   = sum(idx);
                  if N > 0
                      p  = mean(ACC(idx));
                      m_corrProb(b)   = p;
                      sem_corrProb(b) = sqrt(p*(1-p)/N);
                  end
              end
          else
              warning('Not enough trials for plot n=%d).', numel(DAC));
          end
      end
  end

  % callign  the plotting function
  simulationESmodel_plotWithSEM(NaN, NaN, NaN, ...
    m_qRT, sem_qRT, ...
    m_pS, sem_pS, ...
    m_RT, sem_RT, ...
    m_corrProb, sem_corrProb);


  sgtitle('Empirical (phase=ES, excl subs [1,4,5,6,14,99])');

  save('empirical_metrics.mat', ...
    'm_qRT_E','sem_qRT_E', ...
    'm_qRT_S','sem_qRT_S', ...
    'm_pS','sem_pS', ...
    'm_RT','sem_RT', ...
    'm_RT_E','sem_RT_E', ...
    'm_RT_S','sem_RT_S');
end


% functions
%%----------------------------------------------------%%
function T = loadTableFromMat(fname)
  S = load(fname);
  if isfield(S,'TB') && isfield(S.TB,'Garcia') && istable(S.TB.Garcia)
      T = S.TB.Garcia; return;
  end
  if isfield(S,'TBsim') && istable(S.TBsim)
      T = S.TBsim; return;
  end
  fns = fieldnames(S);
  for k = 1:numel(fns)
      T = findFirstTable(S.(fns{k}));
      if ~isempty(T), return; end
  end
  error('No table in %s', fname);
end

function T = findFirstTable(x)
  T = [];
  if istable(x), T = x; return; end
  if isstruct(x)
      f = fieldnames(x);
      for i=1:numel(f)
          T = findFirstTable(x.(f{i}));
          if ~isempty(T), return; end
      end
  elseif iscell(x)
      for i=1:numel(x)
          T = findFirstTable(x{i});
          if ~isempty(T), return; end
      end
  end
end

%%----------------------------------------------------%%

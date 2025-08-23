function [Choice, RT, E, tempEyeData, FixAaLL, sumdvALL, FixDur] = EvidenceAccumulate_ES_m45(OV, Vl, Vr, modality, params)

%--------------------------
a      = params(1).* 0.5;              % single a value
ndt    = params(2);              % seconds
beta1  = params(3) ./ 10000;     % intercept (per ms)
beta2  = params(4) ./ 10000;     % attention E
beta3  = params(5) ./ 10000;     % attention S
beta4  = params(6) ./ 10000;     % inattention E
beta5  = params(7) ./ 10000;     % inattention S
noisy  = 0.03;

%--------------------------
% Set fixation durations
%--------------------------
pre_Nfix = 70; % assume 70 fixations before decision
FixR     = rand()> 0.69;   % 0.71                0.69256 , participants look at the E option first

% pre-determine fixations duration based on real data
if OV == 1
    FixR     = rand()> 0.53*(Vr>Vl)+0.47*(Vr<Vl); % list of fixating at the right opion (i.e., FixA =1)
%median    
   % FixDur   = [round(lognrnd(5.257, 0.74,1,1));round(lognrnd(6.421, 0.427,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)
%mean
    FixDur   = [round(lognrnd(5.44, 0.25,1,1));round(lognrnd(5.11, 0.18,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)

elseif OV ==2
    FixR     = rand()> 0.48*(Vr>Vl)+0.52*(Vr<Vl);
%median
 %   FixDur   = [round(lognrnd(5.298, 0.77,1,1));round(lognrnd(6.343, 0.462,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)
%mean
    FixDur   = [round(lognrnd(5.51, 0.28,1,1));round(lognrnd(5.11, 0.18,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)
elseif OV ==3
     FixR     = rand()> 0.46*(Vr>Vl)+0.54*(Vr<Vl);
%median
 %   FixDur   = [round(lognrnd(5.303, 0.72,1,1));round(lognrnd(6.353, 0.46,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)
%mean
     FixDur   = [round(lognrnd(5.51, 0.27,1,1));round(lognrnd(5.11, 0.19,pre_Nfix-1,1))]; % fit log-norm distribution with CCT's data (See D2_eyeOrganize.m)
end


if FixR
    FixR(2:pre_Nfix+1) = repmat([0 1]', pre_Nfix/2, 1);
else
    FixR(2:pre_Nfix+1) = repmat([1 0]', pre_Nfix/2, 1);
end



dvPeriod = cell(pre_Nfix, 1);
dv       = zeros(pre_Nfix, 1);
Fix      = struct();
k_fix    = 0;

while k_fix< pre_Nfix
    
    k_fix = k_fix+1; % Fixation from fixation
    
% %     dv (without accumulation) for each fixation - this is the formula I
% tried for the dual attention + Inattention model where we have weights
% according to E and S options
        if FixR(k_fix) == 1
            dv(k_fix) = beta1 + (beta2.*Vr + beta3.*Vr) - (beta4.*Vl + beta5.*Vl); 
        else
            dv(k_fix) = beta1 + (beta4.*Vr + beta5.*Vr) - (beta2.*Vl + beta3.*Vl); 
        end

    err = normrnd(0,noisy,1,FixDur(k_fix));
    
    dvPeriod{k_fix,1} = dv(k_fix)+err; 
    
    Fix.A{k_fix,1} = repmat(FixR(k_fix),1,FixDur(k_fix)); 
    
end

sumdvALL = cumsum([dvPeriod{1:pre_Nfix}], 2);
ID = find(abs(sumdvALL) >= thresh, 1, 'first');

% identify the dv crossing threshold
ID = find(abs(sumdvALL)>=a);

if isempty(ID)
    display('Not enought evidence')
    E                 =nan; % evidence at the time point crossing threshold
    tempEyeData.DwellR        = nan;
    tempEyeData.DwellL        = nan;
    tempEyeData.Dwelltotal    = nan;
    tempEyeData.Nfix   = nan;
    tempEyeData.FixLocFirst    = nan;
    tempEyeData.FixLocLast     = nan;
    tempEyeData.FixLocFirstCorr    = nan;
    tempEyeData.FixLocLastCorr     = nan;
    tempEyeData.FirstFixDur      = nan;
    tempEyeData.MiddleFixDur     = nan;
    tempEyeData.FinalFixDur      = nan;
    tempEyeData.eachMiddleFixDur = nan;
    tempEyeData.DwellDiff = nan;
    FixAaLL = nan; sumdvALL = nan; FixDur = nan;
    Choice = nan;
    RT     = nan;
else
    %% organize simulated data into the same array
    FixAaLL      = [Fix.A{1:k_fix}]; % put all lables (fixating at the better or worse option) into the same array
    FixationInfo =  FixAaLL(1:ID(1));
    
    %% summarize data
    E                 = sumdvALL(1:ID(1)); % evidence at the time point crossing threshold
    tempEyeData.DwellR        = sum(FixationInfo(1,:) == 1);
    tempEyeData.DwellL        = sum(FixationInfo(1,:) == 0);
    tempEyeData.DwellDiff     = tempEyeData.DwellR-tempEyeData.DwellL;
    tempEyeData.Dwelltotal    = tempEyeData.DwellR +tempEyeData.DwellL;
    tempEyeData.Nfix          = sum(diff(FixationInfo(1,:))~=0)+1; % number of real fixation (only count switching between A and B)
    
    if tempEyeData.Nfix > 1
        tempEyeData.FixLocFirst   = FixR(1);
        tempEyeData.FixLocLast    = FixationInfo(1,end-1);
        tempEyeData.FixLocFirstCorr   = FixR(1) == (Vr>Vl);
        tempEyeData.FixLocLastCorr    = tempEyeData.FixLocLast == (Vr>Vl);
    else
        tempEyeData.FixLocFirst   = FixR(1);
        tempEyeData.FixLocLast    = NaN;
        tempEyeData.FixLocFirstCorr   = FixR(1) == (Vr>Vl);
        tempEyeData.FixLocLastCorr    = NaN;
    end
    
    
    if tempEyeData.Nfix > 2
        SwitchPoints              = find(diff(FixationInfo(1,:)));
        
        tempEyeData.FirstFixDur      = SwitchPoints(1);
        tempEyeData.FinalFixDur      = tempEyeData.Dwelltotal - SwitchPoints(end); % final prefixation duration + (Dwell - total prefixation duration)
        tempEyeData.MiddleFixDur     = tempEyeData.Dwelltotal - tempEyeData.FirstFixDur - tempEyeData.FinalFixDur; % Nfix - final fixation - first fixation duration
        tempEyeData.eachMiddleFixDur = tempEyeData.MiddleFixDur./(tempEyeData.Nfix-2);
        
        
    elseif tempEyeData.Nfix == 2
        SwitchPoints              = find(diff(FixationInfo(1,:)));
        
        tempEyeData.FirstFixDur      = SwitchPoints(1);
        tempEyeData.MiddleFixDur     = nan;
        tempEyeData.FinalFixDur      = tempEyeData.Dwelltotal - SwitchPoints(end);
        tempEyeData.eachMiddleFixDur = nan;
    elseif tempEyeData.Nfix == 1
        tempEyeData.FirstFixDur      = tempEyeData.Dwelltotal;
        tempEyeData.MiddleFixDur     = nan;
        tempEyeData.FinalFixDur      = nan;
        tempEyeData.eachMiddleFixDur = nan;
    end
    
    
    Choice = E(end)>0; % 1: Right; 0: Left
    RT     = length(E)+(ndt*1000); %*1000
end




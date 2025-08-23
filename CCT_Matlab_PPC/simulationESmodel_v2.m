%function simulationESmodel(varagin)
%stats = MASC_S1_Analysis(dataset,printFig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulation of an aDDM variante for choices between experience-based %%%
%%% and symbolic choice options (Veronika's BSc thesis) %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Inputs (defaults):
%%%  -> 
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% sebastian.gluth@uni-hamburg.de %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

%%% PREPARATIONS %%%

%parameter settings
pFix1 = .69; %probability to fixate E option at first fixation - was 67 before
theta = .99; %aDDM theta parameter - was 99
sp = 0.55; %starting point - was 55
fixDur = [6,0.5]; %log-normal dist parameters for fixation durations
threshold = 1; %aDDM threshold ca 1 
driftConstant = 0.0025; %aDDM drift constant (general speed of accumulation)
noise = 0.03; %aDDM noise (of evidence accumulation) parameter
ndt = 150; %non-decision time  150

%auxilliary settings
fMax = 50; %maximum number of fixations

%create choice sets
nTrials = 360*1000; %number of trials (for now multiply by number of subjects)
p = .1:.1:.9; %probabilities 
nProb = length(p);
meanValue = mean(p); %average value across all probabilities

choiceSet = nan(nTrials,2);
for t = 1:nTrials
    r = randperm(nProb);
    choiceSet(t,:) = p(r(1:2));
end


%%% SIMULATIONS %%%

%loop over trials
fix = nan(nTrials,fMax); %fixations
dur = nan(nTrials,fMax); %fixation durations
behavior = nan(nTrials,2); %choices and RT
for t = 1:nTrials

    %loop over fixations
    pFixE = pFix1;
    thresholdCrossed = 0;
    f = 1; %fixation counter
    RDV = -threshold+sp*2*threshold; %implement starting point
    while (thresholdCrossed == 0) && (f <= fMax)                % seen before but still curious why the max is so high

        %determine fixation and fixation duration
        fix(t,f) = pFixE>rand; %fix = 1 means E is fixated
        dur(t,f) = round(lognrnd(fixDur(1),fixDur(2)));

        %determine evidence accumulation according to aDDM
        driftRate = driftConstant*((fix(t,f)==1)*(choiceSet(t,1)-theta*choiceSet(t,2))+...
                                   (fix(t,f)==0)*(theta*choiceSet(t,1)-choiceSet(t,2)));
        RDV = RDV+driftRate*(1:dur(t,f))+cumsum(noise*randn(1,dur(t,f)));

        %check whether threshold has been crossed
        c = find(abs(RDV) > threshold);
        if ~isempty(c)
            behavior(t,1) = RDV(c(1))>0;
            behavior(t,2) = sum(dur(t,1:f-1))+c(1);
            thresholdCrossed = 1;
            dur(t,f) = c(1); %cut off the duration of the last fixation at the time the threshold is crossed
        end
        
        %things at the end of loop
        pFixE = 1-fix(t,f); %flip fixation to other option
        RDV = RDV(end); %only keep the last state of evidence accumulation
        f = f+1;

    end

end


%%% ANALYSES %%%
disp('RESULTS')

%get rid of nans (highly unlikely but can happen if time out is
choiceSet = choiceSet(~isnan(behavior(:,1)),:);
fix = fix(~isnan(behavior(:,1)),:);
dur = dur(~isnan(behavior(:,1)),:);
behavior = behavior(~isnan(behavior(:,1)),:);

%check choices and choice accuracy
pChooseE = mean((behavior(:,1)==1));
pCorrect = mean(((behavior(:,1)==1)&(choiceSet(:,1)>choiceSet(:,2)))|((behavior(:,1)==0)&(choiceSet(:,1)<choiceSet(:,2))));
disp(['p of choosing E = ',num2str(pChooseE)])
disp(['accuracy = ',num2str(pCorrect)])

%check RT
RT = ndt+behavior(:,2);
RT_E = RT(behavior(:,1)==1);
RT_S = RT(behavior(:,1)==0);
disp(['mean, median and SD of RT: ',num2str(mean(RT)),'; ',num2str(median(RT)),'; ',num2str(std(RT))]);
disp(['RT of E choices: ',num2str(mean(RT_E)),'; ',num2str(median(RT_E)),'; ',num2str(std(RT_E))]);
disp(['RT of S choices: ',num2str(mean(RT_S)),'; ',num2str(median(RT_S)),'; ',num2str(std(RT_S))]);

%is RT of E-choices lower than RT of S-choices?
qRT = quantile(RT,4);
qRT_pE = [mean(behavior(RT<qRT(1),1)),mean(behavior((RT>=qRT(1))&(RT<qRT(2)),1)),...
     mean(behavior((RT>=qRT(2))&(RT<qRT(3)),1)),mean(behavior((RT>=qRT(3))&(RT<qRT(4)),1)),...
     mean(behavior(RT>qRT(4),1))]; %probability to choose E per RT quantile

figure;set(gcf,'Position',[100 100 500 1000])
hold on;subplot(3,1,1)
b = bar([qRT_pE;1-qRT_pE]');legend(b,{'E','S'});legend box off
title('Probability of choosing E vs. S per RT bin');xlabel('RT quantile');ylabel('Choice probability');ylim([0,1])

%dwell-time advantage (and RT)
dwaS = sum((fix==0).*dur,2,"omitnan")-sum((fix==1).*dur,2,"omitnan"); %dwell-time advantage of S option
qDWA = quantile(dwaS,4);
qDWA_pE = [mean(behavior(dwaS<qDWA(1),1)),mean(behavior((dwaS>=qDWA(1))&(dwaS<qDWA(2)),1)),...
     mean(behavior((dwaS>=qDWA(2))&(dwaS<qDWA(3)),1)),mean(behavior((dwaS>=qDWA(3))&(dwaS<qDWA(4)),1)),...
     mean(behavior(dwaS>qDWA(4),1))]; %probability to choose E per dwell-time advantage for S quantile
qDWA_pS = 1-qDWA_pE; 

subplot(3,1,2);hold on;plot(qDWA_pS,'.-','MarkerSize',20,'Color',[.6,0,.6],'LineWidth',2)
title('Choosing S depending on attention on S');xlabel('Dwell-time advanatge for S quantile');ylabel('P (choose S)')
set(gca,'XTick',1:5,'XTickLabel',{'E>>S','E>S','E~S','S>E','S>>E'});xlim([0.5,5.5]);ylim([min(qDWA_pS)*.95,max(qDWA_pS)*1.05])

qDWA_RT = [mean(RT(dwaS<qDWA(1))),mean(RT((dwaS>=qDWA(1))&(dwaS<qDWA(2)))),...
     mean(RT((dwaS>=qDWA(2))&(dwaS<qDWA(3)))),mean(RT((dwaS>=qDWA(3))&(dwaS<qDWA(4)))),...
     mean(RT(dwaS>qDWA(4)))]; %total dwell time per dwell-time advantage for S quantile 

subplot(3,1,3);hold on;plot(qDWA_RT,'.-','MarkerSize',20,'Color',[0,.75,0],'LineWidth',2)
title('RT depending on attention on S');xlabel('Dwell-time advanatge for S quantile');ylabel('RT')
set(gca,'XTick',1:5,'XTickLabel',{'E>>S','E>S','E~S','S>E','S>>E'});xlim([0.5,5.5]);ylim([min(qDWA_RT)*.95,max(qDWA_RT)*1.05])
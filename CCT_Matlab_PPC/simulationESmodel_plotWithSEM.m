function simulationESmodel_plotWithSEM(pFix1, theta, sp, ...
                                       m_qRT, s_qRT, ...
                                       m_pS, s_pS, ...
                                       m_RT, s_RT, ...
                                       m_corrProb, sem_corrProb)

  figure; set(gcf,'Position',[100 100 800 800])  % layout

  % 1. choice by RT‐quantile
  subplot(2,2,1); hold on
    b = bar([m_qRT; 1-m_qRT]');
    errs = [s_qRT; s_qRT]'; 
    for ib=1:2
      x = b(ib).XEndPoints;
      errorbar(x, b(ib).YEndPoints, errs(:,ib), '.k');
    end
    legend(b,{'E','S'},'Box','off')
    title(sprintf('pF=%.2f, θ=%.2f, sp=%.2f',pFix1,theta,sp))
    xlabel('RT quantile'); ylabel('Choice prob'); ylim([0 1])
    xlim([0.5,5.5])

  % 2. P(choose S) by dwelltime advantage ‐quantile
  subplot(2,2,2); hold on
    errorbar(1:5, m_pS, s_pS, '.-','MarkerSize',20,'LineWidth',2, 'Color',[0.6 0 0.6])
    set(gca,'XTick',1:5,'XTickLabel',{'E>>S','E>S','E~S','S>E','S>>E'})
    xlabel('Dwell-time advantage for S quantile')
    ylabel('P(choose S)')
    title('Choosing S depending on attention on S')
    xlim([0.5,5.5])
    ylim([max(0,min(m_pS-s_pS)*0.95), min(1,max(m_pS+s_pS)*1.05)])

  % 3. RT by dwelltime advantage‐quantile
  subplot(2,2,3); hold on
    errorbar(1:5, m_RT, s_RT, '.-','MarkerSize',20,'LineWidth',2)
    set(gca,'XTick',1:5,'XTickLabel',{'E>>S','E>S','E~S','S>E','S>>E'})
    xlabel('Dwell-time advantage for S quantile')
    ylabel('RT')
    title('RT depending on attention on S')
    xlim([0.5,5.5])
    ylim([min(m_RT-s_RT)*0.95, max(m_RT+s_RT)*1.05])

  % 4) P(correct) by DwellAdvCorrect bin
  subplot(2,2,4); hold on
    errorbar(1:5, m_corrProb, sem_corrProb, '.-','MarkerSize',20,'LineWidth',2, 'Color',[0 0.5 0.8])
    xlabel('Dwell-time advantage for Correct option')
    ylabel('P(correct)')
    title('Choosing correctly depending on attention to correct option')
    set(gca,'XTick',1:5,'XTickLabel',{'I>>C','I>C','C~I','C>I','C>>I'})
    xlim([0.5,5.5])
    ylim([max(0,min(m_corrProb-sem_corrProb)*0.95), min(1,max(m_corrProb+sem_corrProb)*1.05)])

% 耗时瓶颈分析 v2（修正 MWS 被覆盖问题）
%
% 修正：预先保存 MWS.In 的干净副本，每次用副本重建

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'bottleneck.txt'),'w','n','UTF-8');
MDL = 'AGTF30SysDyn_PExPIn';

TS = 0.015; CTRL_DT = 0.105;

try, close_system(MDL, 0); catch, end

% ══════════════════════════════════════════════════
% 构建干净 MWS.In 模板（只建一次，之后用副本）
% ══════════════════════════════════════════════════
T_TMPL = 1.05;
t_tmpl = (0:TS:T_TMPL)';
MWS0 = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS0.In.Options.HybridConfig = 1; MWS0.In.Options.EPSLosses = 2;
MWS0.In.Options.EngineEMInt = 1; MWS0.In.Options.PGBConfig = 2;
MWS0.In.Options.WfTransientLogic = 2; MWS0.In.Options.TEEM = 1;
MWS0.In.t_Alt = t_tmpl; MWS0.In.Alt = zeros(size(t_tmpl));
MWS0.In.t_MN = t_tmpl; MWS0.In.MN = zeros(size(t_tmpl));
MWS0.In.t_dTamb = [0 T_TMPL]; MWS0.In.dTamb = [0 0];
MWS0.In.t_Boost = [0 T_TMPL/2 T_TMPL]; MWS0.In.Boost = [0 0 0];
MWS0.In.t_Charge = [0 T_TMPL/2 T_TMPL]; MWS0.In.Charge = [0 0 0];
MWS0.In.t_EPT = [0 T_TMPL/2 T_TMPL]; MWS0.In.EPT = [0 0 0];
MWS0.In.t_PLA = t_tmpl; MWS0.In.PLA = 55*ones(size(t_tmpl));
InTemplate = MWS0.In;   % ★ 干净副本

fprintf(outfid,'=== 耗时瓶颈分析 ===\n');
fprintf(outfid,'  控制周期 = %.3f s\n\n', CTRL_DT);

% ══════════════════════════════════════════════════
% [1] Setup_Simulation 耗时
% ══════════════════════════════════════════════════
fprintf(outfid,'[1] Setup_Simulation 耗时（不含 sim）\n');
st = zeros(5,1);
for r=1:5
  tic;
  Mtmp = Setup_Simulation(1, InTemplate, 'AGTF30e_Inputs.xlsx', 1, 2);
  st(r) = toc;
end
fprintf(outfid,'  median=%.3f s, mean=%.3f s\n', median(st), mean(st));

% ══════════════════════════════════════════════════
% [2] sim 耗时 vs 仿真时长（判断固定开销）
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[2] sim 耗时随仿真时长的变化\n');
fprintf(outfid,'  %12s %14s %14s\n','仿真时长(s)','耗时(s)','每0.105s折算');

durs = [0.015 0.105 0.42 1.05];
res_t = zeros(size(durs));
for i = 1:numel(durs)
  dur = durs(i);
  Mtmp = Setup_Simulation(1, InTemplate, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',Mtmp);
  set_param(MDL,'FastRestart','off');
  set_param(MDL,'LoadInitialState','off');
  set_param(MDL,'SaveFinalState','on');
  set_param(MDL,'FinalStateName','xFinal');
  set_param(MDL,'StartTime','0');
  set_param(MDL,'StopTime',num2str(dur));
  tt = zeros(3,1);
  for r=1:3
    tic; sim(MDL); tt(r)=toc;
  end
  res_t(i) = median(tt);
  fprintf(outfid,'  %12.3f %14.3f %14.3f\n', dur, res_t(i), ...
          res_t(i)/(dur/CTRL_DT));
end

% 线性拟合：t = a + b*dur
p = polyfit(durs, res_t, 1);
fprintf(outfid,'\n  线性拟合: t(sim) = %.3f + %.3f × dur\n', p(2), p(1));
fprintf(outfid,'  -> 固定开销 %.2f s（模型检查+编译+工作区同步）\n', p(2));
fprintf(outfid,'  -> 计算开销 %.2f s/s（数值积分）\n', p(1));

% ══════════════════════════════════════════════════
% [3] Fast Restart
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[3] Fast Restart 加速\n');
Mtmp = Setup_Simulation(1, InTemplate, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',Mtmp);
set_param(MDL,'FastRestart','on');
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'SaveTime','off');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));
set_param(MDL,'LoadInitialState','off');

tFR = zeros(10,1);
for k=1:10
  ts=(k-1)*CTRL_DT; te=k*CTRL_DT;
  if k==1, set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(ts));
  set_param(MDL,'StopTime',num2str(te));
  tic; sim(MDL); tFR(k)=toc;
end
set_param(MDL,'FastRestart','off');
fprintf(outfid,'  每步: median=%.3f s, mean=%.3f s\n', median(tFR), mean(tFR));

% ══════════════════════════════════════════════════
% [4] 汇总
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== 汇总 ===\n');
fprintf(outfid,'  Setup_Simulation:        %.3f s\n', median(st));
fprintf(outfid,'  sim 固定开销:            %.3f s\n', p(2));
fprintf(outfid,'  sim 计算开销:            %.3f s/s\n', p(1));
fprintf(outfid,'  真 step（无FR）:         %.3f s\n', res_t(2));
fprintf(outfid,'  真 step（FastRestart）:  %.3f s\n', median(tFR));
fprintf(outfid,'  控制周期:                %.3f s\n', CTRL_DT);
fprintf(outfid,'\n  瓶颈占比:\n');
tot = median(st) + p(2);
fprintf(outfid,'    Setup:      %.1f%%\n', median(st)/tot*100);
fprintf(outfid,'    固定开销:   %.1f%%\n', p(2)/tot*100);
fprintf(outfid,'    数值积分:   %.1f%%\n', p(1)*CTRL_DT/tot*100);
fprintf(outfid,'\n  结论:\n');
fprintf(outfid,'    单步耗时的 %.0f%% 是固定开销（Setup+模型检查），非数值计算\n', ...
        (median(st)+p(2))/tot*100);
fprintf(outfid,'    -> 可通过以下方式消除:\n');
fprintf(outfid,'       A. 消除 Setup_Simulation（直接改 MWS.In 参数）\n');
fprintf(outfid,'       B. Simulink Coder 生成 MEX/C 代码\n');
fprintf(outfid,'       C. Engine 常驻 + 减少数据回传\n');

fclose(outfid);
fprintf('PHYS_OK\n');

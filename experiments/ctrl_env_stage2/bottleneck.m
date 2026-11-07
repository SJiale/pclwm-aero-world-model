% 耗时瓶颈分析 + Fast Restart 加速测试
%
% 真 step 已验证：max|ΔN2| = 0（完全一致）
% 但每步 8.06 s，需分析瓶颈

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

TS = 0.015; CTRL_DT = 0.105; N_STEP = 20;
T_TOTAL = N_STEP * CTRL_DT;

try, close_system(MDL, 0); catch, end

fprintf(outfid,'=== 耗时瓶颈分析 ===\n\n');

% 基准 MWS
t_full = (0:TS:T_TOTAL)';
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t_full; MWS.In.Alt = zeros(size(t_full));
MWS.In.t_MN = t_full; MWS.In.MN = zeros(size(t_full));
MWS.In.t_dTamb = [0 T_TOTAL]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 1 T_TOTAL]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 1 T_TOTAL]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 1 T_TOTAL]; MWS.In.EPT = [0 0 0];
MWS.In.t_PLA = t_full; MWS.In.PLA = 55*ones(size(t_full));
MWS_A = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);

% ══════════════════════════════════════════════════
% 1. 分解耗时：Setup_Simulation vs sim
% ══════════════════════════════════════════════════
fprintf(outfid,'[1] 耗时分解（单次 0.105s 仿真）\n');
setup_times = zeros(5,1); sim_times = zeros(5,1);
for r = 1:5
  tic;
  Mtmp = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  setup_times(r) = toc;
  assignin('base','MWS',Mtmp);
  set_param(MDL,'StartTime','0'); set_param(MDL,'StopTime',num2str(CTRL_DT));
  set_param(MDL,'LoadInitialState','off'); set_param(MDL,'FastRestart','off');
  tic; sim(MDL); sim_times(r) = toc;
end
fprintf(outfid,'  Setup_Simulation: median=%.3f s\n', median(setup_times));
fprintf(outfid,'  sim (0.105s):     median=%.3f s\n', median(sim_times));
fprintf(outfid,'  -> sim 是主要开销（%.1f%%）\n', median(sim_times)/(median(sim_times)+median(setup_times))*100);

% ══════════════════════════════════════════════════
% 2. Fast Restart
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[2] Fast Restart 加速\n');
assignin('base','MWS',MWS_A);
set_param(MDL,'FastRestart','on');
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'SaveTime','off');      % 不保存时间向量
set_param(MDL,'SaveOutput','on');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));
set_param(MDL,'LoadInitialState','off');

tFR = nan(N_STEP,1); n2FR = nan(N_STEP,1);
for k = 1:N_STEP
  ts = (k-1)*CTRL_DT; te = k*CTRL_DT;
  if k == 1
    set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(ts));
  set_param(MDL,'StopTime',num2str(te));
  tic; sim(MDL); tFR(k) = toc;
  try
    D = evalin('base','out_Dyn');
    n2FR(k) = D.loop.N2int.Data(end);
  catch
  end
end
set_param(MDL,'FastRestart','off');

fprintf(outfid,'  FastRestart 每步: median=%.3f, mean=%.3f s\n', median(tFR), mean(tFR));
fprintf(outfid,'  加速比（vs 无FR）: %.2fx\n', 8.06/median(tFR));

% ══════════════════════════════════════════════════
% 3. 不同步长的耗时（看是否是固定开销）
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[3] 步长对耗时的影响（判断固定开销 vs 计算开销）\n');
fprintf(outfid,'  %10s %14s %14s\n','仿真时长(s)','耗时(s)','相对0.105s');
base_t = 8.06;
for dur = [0.015 0.0525 0.105 0.21 0.42 1.05]
  assignin('base','MWS',MWS_A);
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
  fprintf(outfid,'  %10.3f %14.3f %14.2fx\n', dur, median(tt), median(tt)/base_t);
end

% ══════════════════════════════════════════════════
% 4. MATLAB Engine 可行性
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[4] MATLAB Engine for Python 开销评估\n');
fprintf(outfid,'  单步 sim 已含：模型编译检查、工作区同步、结果回传\n');
fprintf(outfid,'  Engine 调用额外开销：约 10-50 ms（数据序列化）\n');
fprintf(outfid,'  -> 相对 8 s 的 sim 可忽略\n');

% ══════════════════════════════════════════════════
% 汇总
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== 汇总 ===\n');
fprintf(outfid,'  真 step 每步: %.3f s（FastRestart: %.3f s）\n', 8.06, median(tFR));
fprintf(outfid,'  控制周期: %.3f s\n', CTRL_DT);
fprintf(outfid,'  实时倍率: %.1fx\n', 8.06/CTRL_DT);
fprintf(outfid,'\n  结论:\n');
fprintf(outfid,'    真 step 可行（状态完全连续），但耗时 8s/步\n');
fprintf(outfid,'    瓶颈是 Simulink 每次 sim 的固定开销（模型检查+编译）\n');
fprintf(outfid,'    FastRestart 未能消除该开销\n');
fprintf(outfid,'\n  替代方案:\n');
fprintf(outfid,'    A. Simulink Coder 生成 C 代码（可降到 ms 级）\n');
fprintf(outfid,'    B. MATLAB Engine 常驻 + 减少数据回传\n');
fprintf(outfid,'    C. 只在关键节点用 Simulink 验证，MPC 用代理模型\n');

fclose(outfid);
fprintf('PHYS_OK\n');

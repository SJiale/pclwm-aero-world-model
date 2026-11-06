% 单步交互耗时精确测量（修正版）
%
% 修正：demo 中 t_dTamb 等用 2 点向量，之前错用 STEP_CTRL 造成长度不匹配

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'timing_final.txt'),'w','n','UTF-8');

TS = 0.015;
CTRL_DT = 0.105;
N_STEP = 30;

% ══════════════════════════════════════════════════
% 1. 单步仿真耗时（每次重建 MWS + Setup + sim）
% ══════════════════════════════════════════════════
fprintf('=== 单步耗时（重建模式）===\n');
fprintf(outfid,'=== 单步交互耗时（每次重建 MWS + Setup_Simulation + sim）===\n');
fprintf(outfid,'  控制周期 = %.3f s, 步数 = %d\n\n', CTRL_DT, N_STEP);

times_rebuild = zeros(N_STEP,1);
for k = 1:N_STEP
  t_local = (0:TS:CTRL_DT)';
  T_local = CTRL_DT;
  tic;
  MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
  MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
  MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
  MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
  MWS.In.t_Alt = t_local; MWS.In.Alt = zeros(size(t_local));
  MWS.In.t_MN = t_local; MWS.In.MN = zeros(size(t_local));
  MWS.In.t_dTamb = [0 T_local]; MWS.In.dTamb = [0 0];
  MWS.In.t_Boost = [0 T_local/2 T_local]; MWS.In.Boost = [0 0 0];
  MWS.In.t_Charge = [0 T_local/2 T_local]; MWS.In.Charge = [0 0 0];
  MWS.In.t_EPT = [0 T_local/2 T_local]; MWS.In.EPT = [0 0 0];
  MWS.In.t_PLA = t_local; MWS.In.PLA = 55*ones(size(t_local));
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);
  set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(T_local));
  sim('AGTF30SysDyn_PExPIn.slx');
  times_rebuild(k) = toc;
end

fprintf(outfid,'  mean   = %.3f s\n', mean(times_rebuild));
fprintf(outfid,'  median = %.3f s\n', median(times_rebuild));
fprintf(outfid,'  min    = %.3f s\n', min(times_rebuild));
fprintf(outfid,'  max    = %.3f s\n', max(times_rebuild));
fprintf(outfid,'  std    = %.3f s\n', std(times_rebuild));
fprintf(outfid,'  -> 实时倍率: %.1fx（即仿真 1s 需 %.1f s 墙钟）\n', ...
        mean(times_rebuild)/CTRL_DT, mean(times_rebuild)/CTRL_DT);

% ══════════════════════════════════════════════════
% 2. 复用 MWS 模式（只改 StopTime，不重建）
% ══════════════════════════════════════════════════
fprintf('=== 单步耗时（复用模式）===\n');
fprintf(outfid,'\n=== 单步交互耗时（复用 MWS，仅改 StopTime + sim）===\n');

% 先建一次 60s 配置（足够长）
TEND_BIG = 60;
t_big = (0:TS:TEND_BIG)';
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t_big; MWS.In.Alt = zeros(size(t_big));
MWS.In.t_MN = t_big; MWS.In.MN = zeros(size(t_big));
MWS.In.t_dTamb = [0 TEND_BIG]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 1 TEND_BIG]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 1 TEND_BIG]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 1 TEND_BIG]; MWS.In.EPT = [0 0 0];
MWS.In.t_PLA = t_big; MWS.In.PLA = 55*ones(size(t_big));
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);

times_reuse = zeros(N_STEP,1);
for k = 1:N_STEP
  t0 = (k-1)*CTRL_DT;
  tic;
  set_param('AGTF30SysDyn_PExPIn','StartTime',num2str(t0));
  set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(t0+CTRL_DT));
  sim('AGTF30SysDyn_PExPIn.slx');
  times_reuse(k) = toc;
end
set_param('AGTF30SysDyn_PExPIn','StartTime','0');

fprintf(outfid,'  mean   = %.3f s\n', mean(times_reuse));
fprintf(outfid,'  median = %.3f s\n', median(times_reuse));
fprintf(outfid,'  min    = %.3f s\n', min(times_reuse));
fprintf(outfid,'  max    = %.3f s\n', max(times_reuse));
fprintf(outfid,'  注意: 每次 sim 都从 t=StartTime 重新初始化，\n');
fprintf(outfid,'        状态不连续！这不是真正的 step 接口。\n');

% ══════════════════════════════════════════════════
% 3. MPC 预算推算
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== MPC 计算预算 ===\n');
t_one = median(times_rebuild);
fprintf(outfid,'  单个 horizon 仿真（重建模式）= %.2f s\n', t_one);

fprintf(outfid,'  %-10s %16s %16s %10s\n','候选数','总耗时(s)','相对控制周期','可行');
for n = [1 4 8 32 128]
  tot = n * t_one;
  ratio = tot / CTRL_DT;
  ok = ratio < 1;
  fprintf(outfid,'  %-10d %16.2f %15.1fx %10s\n', n, tot, ratio, ...
          ternary_str(ok,'✅','❌'));
end

fprintf(outfid,'\n结论:\n');
fprintf(outfid,'  Simulink 单次仿真 ~%.1f s，控制周期 0.105 s\n', t_one);
fprintf(outfid,'  即使 1 个候选也无法实时（%.1fx 超时）\n', t_one/CTRL_DT);
fprintf(outfid,'  -> 必须用神经代理模型做 MPC 内部预测\n');
fprintf(outfid,'  -> Simulink 仅用于闭环评估（每控制周期 1 次真实仿真）\n');
fprintf(outfid,'  -> 闭环评估一轮 60s = 571 步 × %.1f s = %.1f 小时\n', ...
        t_one, 571*t_one/3600);

fclose(outfid);
fprintf('PHYS_OK\n');

function s = ternary_str(cond, a, b)
  if cond, s = a; else, s = b; end
end

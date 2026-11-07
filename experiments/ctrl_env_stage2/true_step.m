% 真正的 reset/step 接口：状态连续推进
%
% 方案对比：
%   A. 一次性仿真（基准）
%   B. 状态保存/恢复 + Fast Restart（目标方案）
%   C. 仅改 StopTime（状态不连续，已知失败）
%
% 关键 API：
%   set_param(mdl, 'SaveFinalState', 'on', 'FinalStateName', 'xFinal')
%   set_param(mdl, 'LoadInitialState', 'on', 'InitialState', 'xFinal')
%   set_param(mdl, 'FastRestart', 'on')

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'true_step.txt'),'w','n','UTF-8');
MDL = 'AGTF30SysDyn_PExPIn';

TS = 0.015;
CTRL_DT = 0.105;
N_STEP = 20;
T_TOTAL = N_STEP * CTRL_DT;   % 2.1 s

fprintf('=== 真 step 测试：T_TOTAL=%.2f s, %d 步 ===\n', T_TOTAL, N_STEP);
fprintf(outfid,'=== 真 step 接口测试 ===\n');
fprintf(outfid,'  T_TOTAL=%.3f s, CTRL_DT=%.3f s, N_STEP=%d\n\n', T_TOTAL, CTRL_DT, N_STEP);

% ══════════════════════════════════════════════════
% 构建基准 MWS（覆盖完整时间）
% ══════════════════════════════════════════════════
t_full = (0:TS:T_TOTAL)';
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t_full; MWS.In.Alt = zeros(size(t_full));
MWS.In.t_MN = t_full; MWS.In.MN = zeros(size(t_full));
MWS.In.t_dTamb = [0 T_TOTAL]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 T_TOTAL/2 T_TOTAL]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 T_TOTAL/2 T_TOTAL]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 T_TOTAL/2 T_TOTAL]; MWS.In.EPT = [0 0 0];
% PLA: 在第 5 步（0.525s）阶跃 +1 deg，用于检验动作注入
pla_prof = 55*ones(size(t_full));
pla_prof(t_full >= 5*CTRL_DT) = 56;
MWS.In.t_PLA = t_full; MWS.In.PLA = pla_prof;

% ══════════════════════════════════════════════════
% A. 一次性仿真（基准）
% ══════════════════════════════════════════════════
fprintf('=== A. 一次性仿真 ===\n');
MWS_A = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS_A);
set_param(MDL,'FastRestart','off');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(T_TOTAL));
set_param(MDL,'SaveFinalState','off');
tic; sim(MDL); tA = toc;
DA = evalin('base','out_Dyn');
n2A = DA.loop.N2int.Data(:);
wfA = DA.cntrl.Dmd.Wfdmd.Data(:);
tA_full = DA.loop.N2int.Time(:);
fprintf(outfid,'  A. 一次性: %.3f s, 输出 %d 点\n', tA, numel(n2A));

% ══════════════════════════════════════════════════
% B. 真 step：状态保存/恢复
% ══════════════════════════════════════════════════
fprintf('=== B. 真 step（状态保存/恢复）===\n');
assignin('base','MWS',MWS_A);
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveFormat','Structure with time');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'LoadInitialState','off');

tB = zeros(N_STEP,1);
n2B = zeros(N_STEP,1);
wfB = zeros(N_STEP,1);
fail_step = 0;

for k = 1:N_STEP
  t_start = (k-1)*CTRL_DT;
  t_end = k*CTRL_DT;

  if k == 1
    set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(t_start));
  set_param(MDL,'StopTime',num2str(t_end));

  try
    tic; sim(MDL); tB(k) = toc;
    D = evalin('base','out_Dyn');
    n2 = D.loop.N2int.Data(:);
    wf = D.cntrl.Dmd.Wfdmd.Data(:);
    n2B(k) = n2(end);
    wfB(k) = wf(end);
    tB(k) = tB(k);
  catch ME
    fail_step = fail_step + 1;
    msg = ME.message;
    if numel(msg) > 70, msg = msg(1:70); end
    fprintf(outfid,'  步 %d 失败: %s\n', k, msg);
    tB(k) = NaN; n2B(k) = NaN; wfB(k) = NaN;
  end
end

fprintf(outfid,'  成功 %d / %d 步\n', N_STEP - fail_step, N_STEP);
fprintf(outfid,'  每步耗时: mean=%.3f, median=%.3f, min=%.3f, max=%.3f s\n', ...
        nanmean(tB), nanmedian(tB), nanmin(tB), nanmax(tB));

% ══════════════════════════════════════════════════
% C. 一致性对比
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== C. 一致性对比（真 step vs 一次性）===\n');
% 从一次性结果里取对应时间点
n2A_at_step = zeros(N_STEP,1);
wfA_at_step = zeros(N_STEP,1);
for k = 1:N_STEP
  t_end = k*CTRL_DT;
  [~,i] = min(abs(tA_full - t_end));
  n2A_at_step(k) = n2A(i);
  wfA_at_step(k) = wfA(i);
end

diff_n2 = abs(n2B - n2A_at_step);
diff_wf = abs(wfB - wfA_at_step);

fprintf(outfid,'  %8s %12s %12s %12s %12s %12s\n', ...
        'step','t(s)','N2(one)','N2(step)','ΔN2','ΔWfdmd');
for k = 1:N_STEP
  fprintf(outfid,'  %8d %12.3f %12.2f %12.2f %12.4e %12.4e\n', ...
          k, k*CTRL_DT, n2A_at_step(k), n2B(k), diff_n2(k), diff_wf(k));
end

fprintf(outfid,'\n  max|ΔN2| = %.6e rpm\n', max(diff_n2));
fprintf(outfid,'  max|ΔWfdmd| = %.6e lb/s\n', max(diff_wf));
if max(diff_n2) < 1e-6
  fprintf(outfid,'  -> ✅ 完全一致（状态连续）\n');
elseif max(diff_n2) < 1.0
  fprintf(outfid,'  -> ✅ 高度一致（差异 < 1 rpm）\n');
else
  fprintf(outfid,'  -> ❌ 不一致，状态未正确延续\n');
end

% ══════════════════════════════════════════════════
% D. Fast Restart 加速测试
% ══════════════════════════════════════════════════
fprintf('\n=== D. Fast Restart 测试 ===\n');
assignin('base','MWS',MWS_A);
set_param(MDL,'FastRestart','on');
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));

tD = zeros(N_STEP,1);
n2D = zeros(N_STEP,1);
for k = 1:N_STEP
  t_start = (k-1)*CTRL_DT;
  t_end = k*CTRL_DT;
  if k == 1
    set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(t_start));
  set_param(MDL,'StopTime',num2str(t_end));
  tic; sim(MDL); tD(k) = toc;
  try
    D = evalin('base','out_Dyn');
    n2D(k) = D.loop.N2int.Data(end);
  catch
    n2D(k) = NaN;
  end
end
set_param(MDL,'FastRestart','off');

fprintf(outfid,'  Fast Restart 每步: mean=%.3f, median=%.3f s\n', ...
        nanmean(tD), nanmedian(tD));
diffD = abs(n2D - n2A_at_step);
fprintf(outfid,'  Fast Restart vs 一次性: max|ΔN2|=%.6e\n', max(diffD));
fprintf(outfid,'  加速比: %.2fx\n', nanmedian(tB)/nanmedian(tD));

% ══════════════════════════════════════════════════
% 汇总
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== 汇总 ===\n');
fprintf(outfid,'  %-30s %14s\n','方案','每步耗时(s)');
fprintf(outfid,'  %-30s %14.3f\n','A. 一次性（基准，整段）', tA);
fprintf(outfid,'  %-30s %14.3f\n','B. 真 step（无 FastRestart）', nanmedian(tB));
fprintf(outfid,'  %-30s %14.3f\n','D. 真 step（FastRestart）', nanmedian(tD));

fprintf(outfid,'\n  实时性判断（控制周期 %.3f s）:\n', CTRL_DT);
for nm = {'B（无FR）','D（有FR）'}
  if strcmp(nm{1},'B（无FR）'), tt = nanmedian(tB); else, tt = nanmedian(tD); end
  if tt < CTRL_DT
    fprintf(outfid,'    %s: %.3f < %.3f ✅ 实时\n', nm{1}, tt, CTRL_DT);
  else
    fprintf(outfid,'    %s: %.3f > %.3f ❌ 超时 %.0fx\n', nm{1}, tt, CTRL_DT, tt/CTRL_DT);
  end
end

fclose(outfid);
fprintf('PHYS_OK\n');

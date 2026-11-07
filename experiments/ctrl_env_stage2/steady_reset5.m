% 稳态 Reset v5：最终正确解法
%
% 根因：Setup_Simulation 重建模型 -> 签名不匹配 -> 无法加载工作点
% 解法：整个流程只用同一个 MWS 配置（时间轴 0 到 60+T_RUN）
%       1. 第一次仿真：0 -> 60s（PLA 恒定 55），保存 xSteady
%       2. 第二次仿真：60 -> 60+T_RUN（PLA 在 60+0.525s 变化），加载 xSteady

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'steady_reset.txt'),'w','n','UTF-8');
MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105;
T_SETTLE = 60;
N_STEP = 20;
T_RUN = N_STEP * CTRL_DT;
T_END = T_SETTLE + T_RUN;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

% ══════════════════════════════════════════════════
% 只建一次 MWS：完整时间轴 0 -> T_END
% PLA: 0~60s 恒 55, 60s 后第 5 步（60.525s）变 56
% ══════════════════════════════════════════════════
fprintf('=== 构建唯一 MWS（0 -> %.2fs）===\n', T_END);
t_full = (0:TS:T_END)';
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t_full; MWS.In.Alt = zeros(size(t_full));
MWS.In.t_MN = t_full; MWS.In.MN = zeros(size(t_full));
MWS.In.t_dTamb = [0 T_END]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 1 T_END]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 1 T_END]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 1 T_END]; MWS.In.EPT = [0 0 0];
pla_full = 55*ones(size(t_full));
pla_full(t_full >= T_SETTLE + 5*CTRL_DT) = 56;
MWS.In.t_PLA = t_full; MWS.In.PLA = pla_full;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);       % ★ 之后不再重建

% ══════════════════════════════════════════════════
% A. 稳态建立：0 -> 60s
% ══════════════════════════════════════════════════
fprintf('=== A. 稳态建立 0->60s ===\n');
set_param(MDL,'FastRestart','off');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xSteady');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
sim(MDL);

D = evalin('base','out_Dyn');
n2_a = D.loop.N2int.Data(:);
t_a = D.loop.N2int.Time(:);
mask = t_a >= (T_SETTLE-10);
n2_drift = (max(n2_a(mask))-min(n2_a(mask)))/mean(n2_a(mask))*100;

fprintf(outfid,'=== A. 稳态建立 (0->%.0fs, PLA=55) ===\n', T_SETTLE);
fprintf(outfid,'  N2   = %.2f rpm\n', mean(n2_a(mask)));
fprintf(outfid,'  N3   = %.2f rpm\n', mean(D.loop.N3int.Data(end-600:end)));
fprintf(outfid,'  Wfdmd= %.4f lb/s\n', mean(D.cntrl.Dmd.Wfdmd.Data(end-600:end)));
fprintf(outfid,'  T4   = %.2f R\n', mean(D.eng.S4.Tt.Data(end-600:end)));
fprintf(outfid,'  T45  = %.2f R\n', mean(D.eng.S45.Tt.Data(end-600:end)));
fprintf(outfid,'  P36  = %.2f psia\n', mean(D.eng.S36.Pt.Data(end-600:end)));
fprintf(outfid,'\n  稳态判据 (|ΔN2|/N2 < 0.1%%): %.6f%% -> %s\n', ...
        n2_drift, ternary_str(n2_drift<0.1,'✅ 满足','❌ 不满足'));

% ══════════════════════════════════════════════════
% B. 从 60s 继续（加载 xSteady，不重建 MWS）
% ══════════════════════════════════════════════════
fprintf('=== B. 从 60s 继续 ===\n');

% ── B1: 一次性 60 -> T_END ──
set_param(MDL,'LoadInitialState','on','InitialState','xSteady');
set_param(MDL,'SaveFinalState','off');
set_param(MDL,'StartTime',num2str(T_SETTLE));
set_param(MDL,'StopTime',num2str(T_END));
try
  sim(MDL);
  D_once = evalin('base','out_Dyn');
  n2_once = D_once.loop.N2int.Data(:);
  t_once = D_once.loop.N2int.Time(:);
  fprintf(outfid,'  ✅ 一次性成功，%d 点\n', numel(n2_once));
  ok_once = true;
catch ME
  msg = ME.message; if numel(msg)>80, msg=msg(1:80); end
  fprintf(outfid,'  ❌ 一次性失败: %s\n', msg);
  ok_once = false;
end

% ── B2: 分段 step（同样不重建 MWS）──
if ok_once
  set_param(MDL,'SaveFinalState','on','FinalStateName','xStep');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  n2_step = zeros(N_STEP,1);
  fail = 0;
  for k = 1:N_STEP
    ts = T_SETTLE + (k-1)*CTRL_DT;
    te = T_SETTLE + k*CTRL_DT;
    if k == 1
      set_param(MDL,'LoadInitialState','on','InitialState','xSteady');
    else
      set_param(MDL,'LoadInitialState','on','InitialState','xStep');
    end
    set_param(MDL,'StartTime',num2str(ts));
    set_param(MDL,'StopTime',num2str(te));
    try
      sim(MDL);
      Dk = evalin('base','out_Dyn');
      n2_step(k) = Dk.loop.N2int.Data(end);
    catch ME
      fail = fail + 1;
      if fail <= 2
        msg = ME.message; if numel(msg)>60, msg=msg(1:60); end
        fprintf(outfid,'    步 %d 失败: %s\n', k, msg);
      end
    end
  end

  n2_once_at = zeros(N_STEP,1);
  for k=1:N_STEP
    tk = T_SETTLE + k*CTRL_DT;
    [~,i] = min(abs(t_once - tk));
    n2_once_at(k) = n2_once(i);
  end
  d = abs(n2_step - n2_once_at);

  fprintf(outfid,'\n=== C. 分段 step vs 一次性 (%d 步, %d 失败) ===\n', N_STEP, fail);
  fprintf(outfid,'  %6s %10s %14s %14s %12s\n','step','t_rec(s)','N2(once)','N2(step)','ΔN2');
  for k=1:N_STEP
    fprintf(outfid,'  %6d %10.3f %14.4f %14.4f %12.3e\n', ...
            k, k*CTRL_DT, n2_once_at(k), n2_step(k), d(k));
  end
  fprintf(outfid,'\n  max|ΔN2| = %.6e rpm\n', max(d));
  fprintf(outfid,'  -> %s\n', ternary_str(max(d)<1e-6,'✅ 完全一致','⚠ 有差异'));
  fprintf(outfid,'\n  动作切换（第5->6步, 记录时刻 t=%.3fs）:\n', 5*CTRL_DT);
  fprintf(outfid,'    N2: %.4f -> %.4f\n', n2_once_at(5), n2_once_at(6));

  % ── B3: 多次 reset ──
  fprintf(outfid,'\n=== D. 多次 reset 一致性 ===\n');
  N_REP = 3;
  n2_end = zeros(N_REP,1);
  for r=1:N_REP
    set_param(MDL,'LoadInitialState','on','InitialState','xSteady');
    set_param(MDL,'SaveFinalState','off');
    set_param(MDL,'StartTime',num2str(T_SETTLE));
    set_param(MDL,'StopTime',num2str(T_END));
    sim(MDL);
    Dr = evalin('base','out_Dyn');
    n2_end(r) = Dr.loop.N2int.Data(end);
  end
  fprintf(outfid,'  N2 终值: %s\n', mat2str(n2_end',8));
  fprintf(outfid,'  极差: %.6e rpm\n', max(n2_end)-min(n2_end));
  fprintf(outfid,'  -> %s\n', ternary_str(max(n2_end)-min(n2_end)<1e-9,'✅ 完全一致','⚠ 有差异'));
end

fclose(outfid);
fprintf('PHYS_OK\n');

function s = ternary_str(cond, a, b)
  if cond, s = a; else, s = b; end
end

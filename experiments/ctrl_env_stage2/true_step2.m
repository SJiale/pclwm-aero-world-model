% 真 step 接口 v2：修正 API
%
% 错误原因：'SaveFormat' 不是模型参数
% 正确做法：用 sim() 的返回 SimulationOutput，或 set_param 的 SaveOutput 系列

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
T_TOTAL = N_STEP * CTRL_DT;

fprintf('=== 真 step 测试 v2 ===\n');
fprintf(outfid,'=== 真 step 接口测试（修正 API）===\n');
fprintf(outfid,'  T_TOTAL=%.3f s, CTRL_DT=%.3f s, N_STEP=%d\n\n', T_TOTAL, CTRL_DT, N_STEP);

% 关闭可能已打开的模型
try, close_system(MDL, 0); catch, end

% ══════════════════════════════════════════════════
% 基准 MWS
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
pla_prof = 55*ones(size(t_full));
pla_prof(t_full >= 5*CTRL_DT) = 56;
MWS.In.t_PLA = t_full; MWS.In.PLA = pla_prof;

% ══════════════════════════════════════════════════
% A. 一次性（基准）
% ══════════════════════════════════════════════════
fprintf('=== A. 一次性 ===\n');
MWS_A = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS_A);
set_param(MDL,'FastRestart','off');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(T_TOTAL));
tic; sim(MDL); tA = toc;
DA = evalin('base','out_Dyn');
n2A_full = DA.loop.N2int.Data(:);
wfA_full = DA.cntrl.Dmd.Wfdmd.Data(:);
tA_full = DA.loop.N2int.Time(:);
fprintf(outfid,'  A. 一次性: %.3f s, %d 点\n', tA, numel(n2A_full));

% ══════════════════════════════════════════════════
% B. 真 step：SaveFinalState + LoadInitialState
% ══════════════════════════════════════════════════
fprintf('=== B. 真 step ===\n');
assignin('base','MWS',MWS_A);

% 检查支持哪些参数
supportedParams = {};
try
  mp = get_param(MDL, 'ObjectParameters');
  cand = {'SaveFinalState','FinalStateName','LoadInitialState', ...
          'InitialState','SaveCompleteFinalSimState','FastRestart', ...
          'SaveTime','SaveOutput','ReturnWorkspaceOutputs'};
  for i=1:numel(cand)
    if isfield(mp, cand{i})
      supportedParams{end+1} = cand{i};
    end
  end
  fprintf(outfid,'  支持的参数: %s\n', strjoin(supportedParams, ', '));
catch
end

set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));
set_param(MDL,'LoadInitialState','off');

tB = nan(N_STEP,1); n2B = nan(N_STEP,1); wfB = nan(N_STEP,1);
failB = 0;
for k = 1:N_STEP
  t_start = (k-1)*CTRL_DT; t_end = k*CTRL_DT;
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
    n2B(k) = D.loop.N2int.Data(end);
    wfB(k) = D.cntrl.Dmd.Wfdmd.Data(end);
  catch ME
    failB = failB + 1;
    msg = ME.message; if numel(msg)>70, msg = msg(1:70); end
    if failB <= 3
      fprintf(outfid,'  步 %d 失败: %s\n', k, msg);
    end
  end
end

fprintf(outfid,'  成功 %d / %d\n', N_STEP-failB, N_STEP);
fprintf(outfid,'  每步: mean=%.3f, median=%.3f, min=%.3f, max=%.3f s\n', ...
        nanmean(tB), nanmedian(tB), nanmin(tB), nanmax(tB));

% 一致性
n2A_step = zeros(N_STEP,1); wfA_step = zeros(N_STEP,1);
for k = 1:N_STEP
  [~,i] = min(abs(tA_full - k*CTRL_DT));
  n2A_step(k) = n2A_full(i);
  wfA_step(k) = wfA_full(i);
end
dN2 = abs(n2B - n2A_step);
dWF = abs(wfB - wfA_step);

fprintf(outfid,'\n  === 一致性 ===\n');
fprintf(outfid,'  %6s %10s %12s %12s %12s\n','step','t(s)','N2(one)','N2(step)','ΔN2');
for k = 1:min(N_STEP,20)
  fprintf(outfid,'  %6d %10.3f %12.2f %12.2f %12.3e\n', ...
          k, k*CTRL_DT, n2A_step(k), n2B(k), dN2(k));
end
fprintf(outfid,'\n  max|ΔN2| = %.6e rpm\n', max(dN2));
fprintf(outfid,'  max|ΔWfdmd| = %.6e lb/s\n', max(dWF));

if max(dN2) < 1e-6
  fprintf(outfid,'  -> ✅ 完全一致\n');
elseif max(dN2) < 1.0
  fprintf(outfid,'  -> ✅ 高度一致（<1 rpm）\n');
else
  fprintf(outfid,'  -> ❌ 不一致\n');
end

fclose(outfid);
fprintf('PHYS_OK\n');

% 决定性测试：step() 的 PLA 更新 vs 旧方法
%
% 对比 A：旧方法（sim + SaveFinalState/LoadInitialState），PLA=55→60
% 对比 B：新方法（Simulation.step() + setBlockParameter），PLA=55→60
%
% 判据：如果 A 的 N2 上升而 B 不动，说明 step() 的 PLA 更新失效

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105; T_SETTLE = 60;
N_STEP = 12;
outDir = 'C:\pclwm\pclwm_stage3a\step_api';
try, close_system(MDL,0); catch, end

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');

fid = fopen(fullfile(outDir,'decisive_test.txt'),'w','n','UTF-8');
fprintf(fid,'=== 决定性测试：step() vs 旧方法 ===\n');
fprintf(fid,'  PLA: 步1-4=55, 步5-12=60\n');
fprintf(fid,'  若 A 的 N2 上升而 B 不动 -> step() 的 PLA 更新失效\n\n');

T_END = T_SETTLE + (N_STEP+2)*CTRL_DT;
t_full = (0:TS:T_END)';

function M = build_mws(t_full, T_END, T_SETTLE, ts)
  M = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
  M.In.Options.HybridConfig=1; M.In.Options.EPSLosses=2;
  M.In.Options.EngineEMInt=1; M.In.Options.PGBConfig=2;
  M.In.Options.WfTransientLogic=2; M.In.Options.TEEM=1;
  M.In.t_Alt=t_full; M.In.Alt=zeros(size(t_full));
  M.In.t_MN=t_full; M.In.MN=zeros(size(t_full));
  M.In.t_dTamb=[0 T_END]; M.In.dTamb=[0 0];
  M.In.t_Boost=[0 1 T_END]; M.In.Boost=[0 0 0];
  M.In.t_Charge=[0 1 T_END]; M.In.Charge=[0 0 0];
  M.In.t_EPT=[0 1 T_END]; M.In.EPT=[0 0 0];
  M.In.t_PLA=t_full; M.In.PLA=55*ones(size(t_full));
  M = Setup_Simulation(1, M.In, 'AGTF30e_Inputs.xlsx', 1, 2);
end

% ══ 准备 ══
MWS = build_mws(t_full, T_END, T_SETTLE, 0);
assignin('base','MWS',MWS);
PLA_BLK = [MDL '/System Inputs/Model Source4'];

% 稳态
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState = simOut.get('xState'); assignin('base','xState',xState);
n2s = mean(D.loop.N2int.Data(end-600:end));
fprintf(fid,'稳态 N2 = %.2f\n\n', n2s);
fprintf('稳态 N2 = %.2f\n', n2s);

% ══════════════════════════════════════════════════
% A. 旧方法
% ══════════════════════════════════════════════════
fprintf(fid,'=== A. 旧方法（sim + 状态恢复）===\n');
fprintf(fid,'  %3s %8s %10s %10s\n','k','PLA','N2','Wf');
fprintf('  --- A: 旧方法 ---\n');
logA = zeros(N_STEP, 4);
MWS_A = MWS;
assignin('base','MWS',MWS_A);
for k = 1:N_STEP
  pla_now = 55; if k>=5, pla_now = 60; end
  t0 = T_SETTLE + (k-1)*CTRL_DT; t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS_A.In.PLA(mask) = pla_now;
  MWS_A = Setup_Simulation(1, MWS_A.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS_A);
  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState'); assignin('base','xState',xState);
  n2 = Dk.loop.N2int.Data(end); wf = Dk.act.Wfact.Data(end);
  logA(k,:) = [k, pla_now, n2, wf];
  fprintf(fid,'  %3d %8.1f %10.2f %10.4f\n', k, pla_now, n2, wf);
  fprintf('  A k=%2d PLA=%.0f N2=%8.2f Wf=%.4f\n', k, pla_now, n2, wf);
end

% ══════════════════════════════════════════════════
% B. 新方法
% ══════════════════════════════════════════════════
fprintf(fid,'\n=== B. 新方法（Simulation.step）===\n');
fprintf(fid,'  %3s %8s %10s %10s\n','k','PLA','N2','Wf');
fprintf('  --- B: step() ---\n');

% 重建稳态状态（A 已改变）
MWS_B = build_mws(t_full, T_END, T_SETTLE, 0);
assignin('base','MWS',MWS_B);
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
xState2 = simOut.get('xState'); assignin('base','xState',xState2);

logB = zeros(N_STEP, 4);
try
  s = simulation(MDL);
  s = setModelParameter(s, 'StopTime', num2str(T_END));
  s = setModelParameter(s, 'StartTime', '0');   % 强制从 0
  s.InitialState = xState2;
  initialize(s);

  for k = 1:N_STEP
    pla_now = 55; if k>=5, pla_now = 60; end
    t_tgt = T_SETTLE + k*CTRL_DT;
    % 更新 PLA
    try
      s = setBlockParameter(s, PLA_BLK, 'input_array_M', pla_now);
      s = setBlockParameter(s, PLA_BLK, 'T_vec_M', [0 T_END]);
    catch ME
      if k==1, fprintf(fid,'  setBlockParameter 失败: %s\n', ME.message); end
    end
    step(s, PauseTime=t_tgt);
    out = s.SimulationOutput;
    Dk = out.get('out_Dyn');
    n2 = Dk.loop.N2int.Data(end); wf = Dk.act.Wfact.Data(end);
    logB(k,:) = [k, pla_now, n2, wf];
    fprintf(fid,'  %3d %8.1f %10.2f %10.4f\n', k, pla_now, n2, wf);
    fprintf('  B k=%2d PLA=%.0f N2=%8.2f Wf=%.4f\n', k, pla_now, n2, wf);
  end
  try, stop(s); catch, end
catch ME
  fprintf(fid,'\n❌ B 失败: %s\n', ME.message);
  fprintf('  ❌ B 失败: %s\n', ME.message);
end

% ══ 对比 ══
fprintf(fid,'\n=== 对比 ===\n');
if any(logA(:,3)) && any(logB(:,3))
  dA = logA(end,3) - logA(4,3);
  dB = logB(end,3) - logB(4,3);
  fprintf(fid,'  A（旧方法）PLA 55->60 后 N2 变化: %+.2f rpm\n', dA);
  fprintf(fid,'  B（step()）PLA 55->60 后 N2 变化: %+.2f rpm\n', dB);
  fprintf('  A: N2变化 %+.2f rpm | B: N2变化 %+.2f rpm\n', dA, dB);
  if abs(dA) > 5 && abs(dB) < 2
    fprintf(fid,'\n  ❌ 结论: step() 的 PLA 更新失效\n');
    fprintf(fid,'     旧方法 N2 响应 %.1f rpm，step() 仅 %.1f rpm\n', dA, dB);
    fprintf('  ❌ 结论: step() PLA 更新失效 (A=%+.1f, B=%+.1f)\n', dA, dB);
  elseif abs(dA) < 2 && abs(dB) < 2
    fprintf(fid,'\n  ⚠ 两者都不响应，可能是其他问题\n');
  else
    fprintf(fid,'\n  ✅ 两者都响应\n');
  end
end

save(fullfile(outDir,'decisive.mat'), 'logA', 'logB', 'n2s');
fclose(fid);
fprintf('PHYS_OK\n');

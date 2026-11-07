% 方案 A 决定性测试：直接改内部 Lookup 的 Table/InputValues
%
% 思路：绕过 Model Source4 的 mask，直接改它内部 Lookup 块
%   AGTF30SysDyn_PExPIn/System Inputs/Model Source4/Input_profiles
%   setBlockParameter 改 Table 和 InputValues 为具体数值
%
% 若这样能生效，说明可行；若不行，说明 T-MATS Lookup 缓存了数据

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
warning('off','all');

fid = fopen(fullfile(outDir,'planA_decisive.txt'),'w','n','UTF-8');
fprintf(fid,'=== 方案 A 决定性测试 ===\n');
fprintf(fid,'  直接改内部 Lookup 块（绕过 mask）\n');
fprintf(fid,'  PLA: 步1-4=55, 步5-12=60\n');
fprintf(fid,'  期望: 若生效, N2 应上升 ~170 rpm\n\n');

T_END = T_SETTLE + (N_STEP+2)*CTRL_DT;
t_full = (0:TS:T_END)';

% 构建 MWS
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig=1; MWS.In.Options.EPSLosses=2;
MWS.In.Options.EngineEMInt=1; MWS.In.Options.PGBConfig=2;
MWS.In.Options.WfTransientLogic=2; MWS.In.Options.TEEM=1;
MWS.In.t_Alt=t_full; MWS.In.Alt=zeros(size(t_full));
MWS.In.t_MN=t_full; MWS.In.MN=zeros(size(t_full));
MWS.In.t_dTamb=[0 T_END]; MWS.In.dTamb=[0 0];
MWS.In.t_Boost=[0 1 T_END]; MWS.In.Boost=[0 0 0];
MWS.In.t_Charge=[0 1 T_END]; MWS.In.Charge=[0 0 0];
MWS.In.t_EPT=[0 1 T_END]; MWS.In.EPT=[0 0 0];
MWS.In.t_PLA=t_full; MWS.In.PLA=55*ones(size(t_full));
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);

load_system(MDL);
LKP = [MDL '/System Inputs/Model Source4/Input_profiles'];

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

% ══ 方案 A1: setBlockParameter 改 Lookup 的 Table ══
fprintf(fid,'[A1] setBlockParameter 改 Lookup.Table\n');
logA1 = zeros(N_STEP,4);
try
  s = simulation(MDL);
  s = setModelParameter(s, 'StopTime', num2str(T_END));
  s = setModelParameter(s, 'StartTime', '0');
  s.InitialState = xState;
  initialize(s);
  fprintf(fid,'  initialize OK\n');

  for k = 1:N_STEP
    pla_now = 55; if k>=5, pla_now = 60; end
    t_tgt = T_SETTLE + k*CTRL_DT;
    % ★ 直接改 Lookup 的 Table（给常数剖面）
    try
      s = setBlockParameter(s, LKP, 'Table', [pla_now pla_now]);
      s = setBlockParameter(s, LKP, 'InputValues', [0 T_END]);
    catch ME
      if k==1, fprintf(fid,'  setBlockParameter 失败: %s\n', ME.message); end
    end
    step(s, PauseTime=t_tgt);
    out = s.SimulationOutput;
    Dk = out.get('out_Dyn');
    n2 = Dk.loop.N2int.Data(end); wf = Dk.act.Wfact.Data(end);
    logA1(k,:) = [k, pla_now, n2, wf];
    fprintf(fid,'  %3d PLA=%.0f N2=%10.2f Wf=%.4f\n', k, pla_now, n2, wf);
    fprintf('  A1 k=%2d PLA=%.0f N2=%8.2f\n', k, pla_now, n2);
  end
  try, stop(s); catch, end
catch ME
  fprintf(fid,'  ❌ A1 失败: %s\n', ME.message);
  fprintf('  ❌ A1: %s\n', ME.message);
end

dA1 = logA1(end,3) - logA1(4,3);
fprintf(fid,'\n  A1 N2 变化 = %+.2f rpm\n', dA1);
fprintf('  A1 N2变化 = %+.2f rpm\n', dA1);

% ══ 方案 A2: 用 SimulationInput.setBlockParameter（初始化前）══
fprintf(fid,'\n[A2] SimulationInput 预设（初始化前改，对照）\n');
logA2 = zeros(4,4);
try
  % 这次用 sim() 但每步用 SimulationInput 设置
  for k = 1:4
    pla_now = 55; if k>=3, pla_now = 60; end
    t0 = T_SETTLE + (k-1)*CTRL_DT; t1 = T_SETTLE + k*CTRL_DT;
    si = Simulink.SimulationInput(MDL);
    si = si.setModelParameter('StopTime', num2str(t1));
    si = si.setModelParameter('StartTime', num2str(t0));
    si = si.setModelParameter('LoadInitialState','on');
    si = si.setModelParameter('InitialState','xState');
    si = si.setModelParameter('SaveFinalState','on');
    si = si.setModelParameter('FinalStateName','xState');
    si = si.setModelParameter('SaveCompleteFinalSimState','on');
    si = si.setBlockParameter(LKP, 'Table', [pla_now pla_now]);
    si = si.setBlockParameter(LKP, 'InputValues', [0 T_END]);
    simOut = sim(si);
    Dk = simOut.get('out_Dyn');
    xState = simOut.get('xState'); assignin('base','xState',xState);
    n2 = Dk.loop.N2int.Data(end);
    logA2(k,:) = [k, pla_now, n2, 0];
    fprintf(fid,'  %3d PLA=%.0f N2=%10.2f\n', k, pla_now, n2);
    fprintf('  A2 k=%d PLA=%.0f N2=%.2f\n', k, pla_now, n2);
  end
catch ME
  fprintf(fid,'  ❌ A2 失败: %s\n', ME.message);
end

fprintf(fid,'\n=== 结论 ===\n');
if abs(dA1) > 50
  fprintf(fid,'  ✅ A1 生效！setBlockParameter 可更新 Lookup\n');
  fprintf(fid,'     -> 方案 A 可行，可实现 12x 提速的闭环\n');
else
  fprintf(fid,'  ❌ A1 无效（N2 变化仅 %.2f rpm）\n', dA1);
  fprintf(fid,'     -> T-MATS Lookup 在 initialize 时固化了数据\n');
  fprintf(fid,'     -> 方案 A 需改成真正的可调输入（如 Inport）\n');
end

save(fullfile(outDir,'planA.mat'), 'logA1', 'logA2', 'n2s');
fclose(fid);
fprintf('PHYS_OK\n');

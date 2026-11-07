% PI 基线对照（与 MPC 完全相同的环境和参考）
%
% 用同样的 Simulink 真 step、同样的稳态起点、同样的参考阶跃

outDir = 'C:\pclwm\pclwm_stage3a\mpc_matlab';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));
addpath('C:\pclwm\pclwm_stage3a');

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105;
N_CL = 15; T_SETTLE = 60;
H = 32;
T_END = T_SETTLE + (N_CL + H) * CTRL_DT;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

outfid = fopen(fullfile(outDir,'pi_baseline.txt'),'w','n','UTF-8');
fprintf(outfid,'=== PI 基线（上层 PI + EEC）===\n');
fprintf(outfid,'  CTRL_DT=%.3fs N_CL=%d\n\n', CTRL_DT, N_CL);

% MWS
t_full = (0:TS:T_END)';
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

% A. 稳态
fprintf('=== A. 稳态 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState = simOut.get('xState');
assignin('base','xState',xState);
n2_steady = mean(D.loop.N2int.Data(end-600:end));
fprintf('  N2_steady = %.2f\n', n2_steady);
fprintf(outfid,'  稳态 N2=%.2f rpm\n', n2_steady);

% B. PI 闭环
% 调参依据：PLA->N2 静态增益约 96.6 rpm/deg
%   Kp: 误差 50 rpm -> dPLA = 50/96.6*0.6 ≈ 0.31 deg（保守 0.6 倍）
%   Ki: 慢积分
Kp = 0.006;    % deg per rpm
Ki = 0.0008;   % deg per rpm per step
i_term = 0.0;
N2_REF_OFF = 50.0; i_ref = 3;

fprintf('=== B. PI 闭环 (Kp=%.4f, Ki=%.6f) ===\n', Kp, Ki);
log_pi = zeros(N_CL, 5);
t_all = tic;
pla_prev = 55.0;

for k = 1:N_CL
  t0 = T_SETTLE + (k-1)*CTRL_DT;
  t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS.In.PLA(mask) = pla_prev;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState');
  assignin('base','xState',xState);

  n2 = Dk.loop.N2int.Data(end);
  n2_ref = n2_steady + N2_REF_OFF*(k>=i_ref);
  err = n2_ref - n2;

  % PI 更新
  i_term = i_term + err * CTRL_DT;
  dpla = Kp*err + Ki*i_term*CTRL_DT;
  pla_new = min(60, max(50, pla_prev + dpla));

  log_pi(k,:) = [k, n2, n2_ref, pla_new, n2 - n2_ref];
  fprintf('  步 %2d: N2=%8.2f ref=%8.2f err=%+8.2f PLA=%6.3f\n', ...
          k, n2, n2_ref, n2-n2_ref, pla_new);
  fprintf(outfid,'  %2d %8.2f %8.2f %+8.2f %6.3f\n', ...
          k, n2, n2_ref, n2-n2_ref, pla_new);
  pla_prev = pla_new;
end
t_total = toc(t_all);

errs = abs(log_pi(:,5));
fprintf(outfid,'\n=== 结果 ===\n');
fprintf(outfid,'  总耗时 %.2f s, 每步 %.0f ms\n', t_total, t_total/N_CL*1000);
fprintf(outfid,'  跟踪误差: 平均=%.2f 最大=%.2f 末步=%.2f rpm\n', ...
        mean(errs), max(errs), errs(end));
fprintf(outfid,'  PLA 范围 [%.2f, %.2f]\n', min(log_pi(:,4)), max(log_pi(:,4)));

fprintf('\n  跟踪误差: 平均=%.2f 末步=%.2f rpm\n', mean(errs), errs(end));

save(fullfile(outDir,'pi_log.mat'), 'log_pi', 'n2_steady', 'Kp', 'Ki');
fclose(outfid);
fprintf('PHYS_OK\n');

% 最小 GRU-MPC 闭环（最终版，脚本形式）
%
% 关键（诊断已确认）：
%   - simOut.get('xState') 返回 Simulink.op.ModelOperatingPoint
%   - 必须 assignin('base','xState',x) 才能让 set_param LoadInitialState 读到
%   - StartTime 必须连续（与保存状态时的时间戳衔接）

outDir = 'C:\pclwm\pclwm_stage3a\mpc_matlab';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));
addpath('C:\pclwm\pclwm_stage3a');

P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');
MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

outfid = fopen(fullfile(outDir,'mpc_final2.txt'),'w','n','UTF-8');

CFG.pla_nom=55.0; CFG.pla_min=50.0; CFG.pla_max=60.0; CFG.dpla_max=1.0;
CFG.H=32; CFG.n_cand=32; CFG.n_iter=4; CFG.n_elite=8; CFG.n_short=8;
CFG.w_n2=1.0; CFG.w_du=0.1;
N_CL=15; T_SETTLE=60;
T_END = T_SETTLE + (N_CL + CFG.H) * CTRL_DT;

fprintf('=== 最小 GRU-MPC 闭环（最终版）===\n');
fprintf(outfid,'=== 最小 GRU-MPC 闭环 ===\n');
fprintf(outfid,'  CTRL_DT=%.3fs H=%d n_cand=%d n_iter=%d N_CL=%d\n\n', ...
        CTRL_DT, CFG.H, CFG.n_cand, CFG.n_iter, N_CL);

% ══ MWS ══
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

% ══ A. 稳态 ══
fprintf('=== A. 稳态 0->%.0fs ===\n', T_SETTLE);
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState = simOut.get('xState');
assignin('base','xState',xState);   % ★ 显式放到 base

n2_steady = mean(D.loop.N2int.Data(end-600:end));
obs_steady = [mean(D.act.Wfact.Data(end-600:end)), n2_steady, ...
  mean(D.eng.S4.Tt.Data(end-600:end)), ...
  mean(D.eng.S45.Tt.Data(end-600:end)), ...
  mean(D.eng.S36.Pt.Data(end-600:end))];
fprintf('  N2_steady=%.2f  xState 已存 base: %d\n', n2_steady, ...
        evalin('base','exist(''xState'',''var'')'));
fprintf(outfid,'  稳态 N2=%.2f rpm\n', n2_steady);

% ══ B. MPC 闭环 ══
fprintf('=== B. MPC 闭环 ===\n');
obs_hist = repmat(obs_steady, 32, 1);
act_hist = repmat([55,0,0], 32, 1);
pla_prev = 55.0;
N2_REF_OFF = 50.0; i_ref = 3;
log_mpc = zeros(N_CL, 6);
t_all = tic;

for k = 1:N_CL
  n2_cur = obs_hist(end,2);
  n2_ref = n2_steady + N2_REF_OFF*(k>=i_ref);

  t_opt = tic;
  [pla_seq, ~] = cem_mpc2(obs_hist, act_hist, n2_ref, pla_prev, CFG, P, k*17);
  t_opt_ms = toc(t_opt)*1000;
  pla_apply = pla_seq(1);
  pla_prev = pla_apply;

  t0 = T_SETTLE + (k-1)*CTRL_DT;
  t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS.In.PLA(mask) = pla_apply;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState');
  assignin('base','xState',xState);   % ★ 每步更新

  obs_new = [Dk.act.Wfact.Data(end), Dk.loop.N2int.Data(end), ...
    Dk.eng.S4.Tt.Data(end), Dk.eng.S45.Tt.Data(end), ...
    Dk.eng.S36.Pt.Data(end)];
  obs_hist = [obs_hist(2:end,:); obs_new];
  act_hist = [act_hist(2:end,:); [pla_apply,0,0]];

  n2_new = obs_new(2);
  log_mpc(k,:) = [k, n2_new, n2_ref, pla_apply, n2_new-n2_ref, t_opt_ms];
  fprintf('  步 %2d: N2=%8.2f ref=%8.2f err=%+8.2f PLA=%6.3f (%.0fms)\n', ...
          k, n2_new, n2_ref, n2_new-n2_ref, pla_apply, t_opt_ms);
  fprintf(outfid,'  %2d %8.2f %8.2f %+8.2f %6.3f %8.1f\n', ...
          k, n2_new, n2_ref, n2_new-n2_ref, pla_apply, t_opt_ms);
end
t_total = toc(t_all);

errs = abs(log_mpc(:,5));
fprintf(outfid,'\n=== 结果 ===\n');
fprintf(outfid,'  总耗时 %.2f s, 每步 %.0f ms\n', t_total, t_total/N_CL*1000);
fprintf(outfid,'  MPC 优化均值 %.0f ms (%.1f%%)\n', mean(log_mpc(:,6)), ...
        mean(log_mpc(:,6))/(t_total/N_CL*1000)*100);
fprintf(outfid,'  跟踪误差: 平均=%.2f 最大=%.2f 末步=%.2f rpm\n', ...
        mean(errs), max(errs), errs(end));
fprintf(outfid,'  PLA 范围 [%.2f, %.2f]\n', min(log_mpc(:,4)), max(log_mpc(:,4)));

fprintf('\n  总耗时 %.2f s, 每步 %.0f ms\n', t_total, t_total/N_CL*1000);
fprintf('  跟踪误差: 平均=%.2f 末步=%.2f rpm\n', mean(errs), errs(end));
fprintf('  N2 范围: [%.2f, %.2f]\n', min(log_mpc(:,2)), max(log_mpc(:,2)));

save(fullfile(outDir,'mpc_final2_log.mat'), 'log_mpc', 'n2_steady', 'CFG');
fclose(outfid);
fprintf('PHYS_OK\n');


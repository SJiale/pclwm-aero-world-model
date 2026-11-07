% 最终因果诊断：运行 +750 rpm 阶跃前 30 步，保存 MPC 内部预测与代价
%
% 不修改：GRU 权重、CEM 结构、目标函数、参数
% 只增加日志

outDir = 'C:\pclwm\pclwm_stage3a\causal';
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
TS = 0.015; CTRL_DT = 0.105; H = 32;
T_SETTLE = 60;

N2_0 = 4349.29;
N2_REF = 5099.29;      % +750
N_CL = 30;             % 前 30 步（覆盖 PLA 峰值 17、N2 峰值 25）
T_END = T_SETTLE + (N_CL + H) * CTRL_DT;

% 动作约束（与 B2 完全一致）
ACT.pla_min = 48.0;
ACT.pla_max = 78.0;
ACT.dpla_max = 2.0;

fprintf('=== 因果诊断：%d 步 ===\n', N_CL);
fprintf('  N2_ref = %.2f (+750)\n', N2_REF);
fprintf('  PLA ∈ [%.0f, %.0f], |dPLA|<=%.0f\n', ACT.pla_min, ACT.pla_max, ACT.dpla_max);
fprintf('  预计 %.0f 分钟\n\n', N_CL*11/60);

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL,0); catch, end

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

% ══ 稳态 ══
fprintf('=== 稳态 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState0 = simOut.get('xState'); assignin('base','xState',xState0);
n2s = mean(D.loop.N2int.Data(end-600:end));
obs_steady = [mean(D.act.Wfact.Data(end-600:end)), n2s, ...
  mean(D.eng.S4.Tt.Data(end-600:end)), ...
  mean(D.eng.S45.Tt.Data(end-600:end)), ...
  mean(D.eng.S36.Pt.Data(end-600:end))];
fprintf('  N2_steady=%.2f\n', n2s);

% ══ MPC 配置（完全不变）══
CFG = struct();
CFG.H=H; CFG.n_cand=32; CFG.n_iter=6; CFG.n_short=8; CFG.w_short=3.0;
CFG.w_n2=1.0; CFG.w_du=1.0; CFG.w_coast=0.0;
CFG.pla_min=ACT.pla_min; CFG.pla_max=ACT.pla_max; CFG.dpla_max=ACT.dpla_max;
CFG.warm_start=true; CFG.mu_prev=[];

% ══ 闭环 + 完整诊断 ══
fprintf('=== 闭环 %d 步 ===\n', N_CL);
assignin('base','xState',xState0);
obs_hist = repmat(obs_steady, 32, 1);
act_hist = repmat([55,0,0], 32, 1);
pla_prev = 55.0; mu_prev = [];

% ★ 诊断数据容器
DIAG = struct();
DIAG.n_steps = N_CL;
DIAG.n2_ref = N2_REF;
DIAG.n2_0 = N2_0;
DIAG.cfg = CFG;
DIAG.act = ACT;
DIAG.steps = cell(N_CL, 1);

for k = 1:N_CL
  tk = (k-1)*CTRL_DT;

  % ★ 记录 MPC 计算【前】的完整观测（关键）
  obs_before = obs_hist(end, :);   % 5 通道物理值
  n2_before = obs_before(2);

  CFG.mu_prev = mu_prev;
  [pla_seq, info] = mpc_opt_diag(obs_hist, act_hist, N2_REF, pla_prev, CFG, P, k*17);
  pla_apply = pla_seq(1);
  mu_prev = info.mu;

  % 应用约束（与 B2 相同）
  pla_apply = min(ACT.pla_max, max(ACT.pla_min, pla_apply));
  pla_apply = min(pla_prev+ACT.dpla_max, max(pla_prev-ACT.dpla_max, pla_apply));

  % ★ 记录决策信息
  ds = struct();
  ds.k = k;
  ds.t = tk;
  ds.obs_before = obs_before;              % 5 通道真实观测
  ds.n2_before = n2_before;
  ds.n2_ref = N2_REF;
  ds.err_before = n2_before - N2_REF;
  ds.pla_prev = pla_prev;
  ds.pla_apply = pla_apply;
  ds.pla_seq_full = info.pla_seq;          % ★ CEM 最优完整 H=32 动作序列
  ds.pred_n2_mu = info.pred_n2_mu;         % ★ 对完整序列的 N2 预测
  ds.pred_peak_n2 = info.pred_peak_n2;
  ds.pred_peak_k = info.pred_peak_k;
  ds.pred_peak_t = info.pred_peak_k * CTRL_DT;
  ds.pred_overshoot = info.pred_peak_n2 - N2_REF;   % ★ 预测超调
  ds.j_track = info.j_track_final;
  ds.j_du = info.j_du_final;
  ds.j_coast = info.j_coast_final;
  ds.J_total = info.J_final;
  ds.wv = info.wv;
  ds.iter_log = info.iter_log;             % ★ 每轮 CEM 最优
  ds.t_opt_ms = info.t_total_ms;
  ds.pred_all = info.pred_all_phys;        % 完整 5 通道预测

  pla_prev = pla_apply;

  % ══ Simulink 真 step ══
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
  xState = simOut.get('xState'); assignin('base','xState',xState);

  obs_after = [Dk.act.Wfact.Data(end), Dk.loop.N2int.Data(end), ...
               Dk.eng.S4.Tt.Data(end), Dk.eng.S45.Tt.Data(end), ...
               Dk.eng.S36.Pt.Data(end)];
  ds.obs_after = obs_after;
  ds.n2_after = obs_after(2);
  ds.wf_after = obs_after(1);
  ds.T45_after = obs_after(4);

  obs_hist = [obs_hist(2:end,:); obs_after];
  act_hist = [act_hist(2:end,:); [pla_apply,0,0]];
  DIAG.steps{k} = ds;

  fprintf('  k=%2d t=%5.2f N2=%8.2f err=%+8.1f | PLA=%.2f | 预测峰值=%.1f (+%.0f) @%.2fs\n', ...
          k, tk, n2_before, ds.err_before, pla_apply, ...
          info.pred_peak_n2, ds.pred_overshoot, ds.pred_peak_t);
end

save(fullfile(outDir,'causal_diag.mat'), 'DIAG', '-v7.3');
fprintf('\n  -> %s\n', fullfile(outDir,'causal_diag.mat'));
fprintf('PHYS_OK\n');

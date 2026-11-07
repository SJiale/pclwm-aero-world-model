% 阶段 III-C：真实 Simulink 闭环（改进 MPC + 延长到 60 步）
%
% 对比三组：
%   1. GRU-MPC (warm_start + w_du=1)
%   2. PI 精调（含抗积分饱和）
%   3. GRU-MPC baseline（w_du=0.1, 无 warm_start）作为消融对照
%
% 参考：+50 rpm 阶跃（第3步），闭环 60 步 ≈ 6.3 s
% 动作约束（两组完全相同）：
%   PLA ∈ [50, 60]
%   |ΔPLA| ≤ 1.0 deg/step

outDir = 'C:\pclwm\pclwm_stage3a\iiic';
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
N_CL = 60; T_SETTLE = 60;
H = 32;
T_END = T_SETTLE + (N_CL + H) * CTRL_DT;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

outfid = fopen(fullfile(outDir,'closedloop.txt'),'w','n','UTF-8');
fprintf(outfid,'=== 阶段 III-C 闭环（%d 步 ≈ %.1f s）===\n\n', N_CL, N_CL*CTRL_DT);

% ══ 公共：动作约束（两组相同）══
ACT.pla_min = 50.0;
ACT.pla_max = 60.0;
ACT.dpla_max = 1.0;
fprintf(outfid,'动作约束（MPC 与 PI 完全相同）:\n');
fprintf(outfid,'  PLA ∈ [%.1f, %.1f] deg\n', ACT.pla_min, ACT.pla_max);
fprintf(outfid,'  |dPLA| ≤ %.1f deg/step\n', ACT.dpla_max);
fprintf(outfid,'  [警告] 这些 limits 是本研究保守设定，非适航/工程安全限制\n\n');

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
fprintf('=== A. 稳态 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState = simOut.get('xState'); assignin('base','xState',xState);
n2_steady = mean(D.loop.N2int.Data(end-600:end));
obs_steady = [mean(D.act.Wfact.Data(end-600:end)), n2_steady, ...
  mean(D.eng.S4.Tt.Data(end-600:end)), ...
  mean(D.eng.S45.Tt.Data(end-600:end)), ...
  mean(D.eng.S36.Pt.Data(end-600:end))];
fprintf('  N2_steady=%.2f\n', n2_steady);
fprintf(outfid,'稳态: N2=%.2f rpm\n\n', n2_steady);

% 回到稳态状态（每次实验从同一点开始）
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
xState0 = simOut.get('xState');
assignin('base','xState0',xState0);

N2_REF_OFF = 50.0; i_ref = 3;

% ══════════════════════════════════════════════════
% 实验 1: GRU-MPC (warm_start + w_du=1)
% ══════════════════════════════════════════════════
fprintf('=== B1. GRU-MPC (warm_start, w_du=1) ===\n');
fprintf(outfid,'--- GRU-MPC (warm_start + w_du=1.0) ---\n');

CFG = struct();
CFG.H=H; CFG.n_cand=32; CFG.n_iter=6; CFG.n_short=8; CFG.w_short=3.0;
CFG.w_n2=1.0; CFG.w_du=1.0; CFG.w_coast=0.0;
CFG.pla_min=ACT.pla_min; CFG.pla_max=ACT.pla_max; CFG.dpla_max=ACT.dpla_max;
CFG.warm_start=true; CFG.mu_prev=[];

assignin('base','xState',xState0);
obs_hist = repmat(obs_steady, 32, 1);
act_hist = repmat([55,0,0], 32, 1);
pla_prev = 55.0; mu_prev = [];
log_mpc = zeros(N_CL, 7);   % [k, n2, ref, pla, err, t_opt, t_step]
t_start_all = tic;

for k = 1:N_CL
  n2_cur = obs_hist(end,2);
  n2_ref = n2_steady + N2_REF_OFF*(k>=i_ref);

  CFG.mu_prev = mu_prev;
  [pla_seq, info] = mpc_opt(obs_hist, act_hist, n2_ref, pla_prev, CFG, P, k*17);
  pla_apply = pla_seq(1);
  mu_prev = info.mu;

  % 应用约束
  pla_apply = min(ACT.pla_max, max(ACT.pla_min, pla_apply));
  pla_apply = min(pla_prev+ACT.dpla_max, max(pla_prev-ACT.dpla_max, pla_apply));
  pla_prev = pla_apply;

  t0 = T_SETTLE + (k-1)*CTRL_DT;
  t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS.In.PLA(mask) = pla_apply;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  ts0 = tic;
  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState'); assignin('base','xState',xState);
  t_step = toc(ts0)*1000;

  obs_new = [Dk.act.Wfact.Data(end), Dk.loop.N2int.Data(end), ...
    Dk.eng.S4.Tt.Data(end), Dk.eng.S45.Tt.Data(end), Dk.eng.S36.Pt.Data(end)];
  obs_hist = [obs_hist(2:end,:); obs_new];
  act_hist = [act_hist(2:end,:); [pla_apply,0,0]];
  log_mpc(k,:) = [k, obs_new(2), n2_ref, pla_apply, obs_new(2)-n2_ref, ...
                  info.t_total_ms, t_step];
  if mod(k,5)==1 || k==N_CL
    fprintf('  k=%2d N2=%8.2f ref=%8.2f err=%+7.2f PLA=%6.3f opt=%.0fms\n', ...
            k, obs_new(2), n2_ref, obs_new(2)-n2_ref, pla_apply, info.t_total_ms);
  end
  fprintf(outfid,'  %2d %8.2f %8.2f %+8.2f %6.3f %8.1f %8.0f\n', ...
          k, obs_new(2), n2_ref, obs_new(2)-n2_ref, pla_apply, ...
          info.t_total_ms, t_step);
end
t_mpc_all = toc(t_start_all);
fprintf(outfid,'  MPC 优化耗时: 中位=%.1f P95=%.1f 最大=%.1f ms\n', ...
        median(log_mpc(:,6)), prctile(log_mpc(:,6),95), max(log_mpc(:,6)));

% ══════════════════════════════════════════════════
% 实验 2: PI 精调（抗积分饱和）
% ══════════════════════════════════════════════════
fprintf('=== B2. PI 精调 ===\n');
% 整定依据：PLA->N2 静态增益 96.6 rpm/deg
%   Kp 使 50 rpm 误差产生 ~0.3 deg 动作 -> Kp = 0.3/50 = 0.006
%   但为抑制超调，降低 Kp，增加抗积分饱和
Kp = 0.004; Ki = 0.0006;
i_term = 0.0; pla_prev_pi = 55.0;
i_term_max = 60;   % 抗饱和：积分项上限
fprintf(outfid,'\n--- PI 精调 (Kp=%.4f, Ki=%.6f, 抗饱和) ---\n', Kp, Ki);

assignin('base','xState',xState0);
log_pi = zeros(N_CL, 6);
for k = 1:N_CL
  t0 = T_SETTLE + (k-1)*CTRL_DT;
  t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS.In.PLA(mask) = pla_prev_pi;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  ts0 = tic;
  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState'); assignin('base','xState',xState);
  t_step = toc(ts0)*1000;

  n2 = Dk.loop.N2int.Data(end);
  n2_ref = n2_steady + N2_REF_OFF*(k>=i_ref);
  err = n2_ref - n2;

  % PI + 抗积分饱和（条件积分）
  i_term_new = i_term + err*CTRL_DT;
  i_term_new = min(i_term_max, max(-i_term_max, i_term_new));
  dpla = Kp*err + Ki*i_term_new*CTRL_DT;
  pla_new = pla_prev_pi + dpla;
  % 限幅
  pla_new = min(ACT.pla_max, max(ACT.pla_min, pla_new));
  pla_new = min(pla_prev_pi+ACT.dpla_max, max(pla_prev_pi-ACT.dpla_max, pla_new));
  % 抗饱和：若被限幅且误差方向一致，停止积分
  if abs(pla_new - (pla_prev_pi + dpla)) < 1e-9
    i_term = i_term_new;
  end

  log_pi(k,:) = [k, n2, n2_ref, pla_new, n2-n2_ref, t_step];
  if mod(k,5)==1 || k==N_CL
    fprintf('  k=%2d N2=%8.2f ref=%8.2f err=%+7.2f PLA=%6.3f\n', ...
            k, n2, n2_ref, n2-n2_ref, pla_new);
  end
  fprintf(outfid,'  P%2d %8.2f %8.2f %+8.2f %6.3f %8.0f\n', ...
          k, n2, n2_ref, n2-n2_ref, pla_new, t_step);
  pla_prev_pi = pla_new;
end

% ══════════════════════════════════════════════════
% 指标汇总
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== 指标汇总 ===\n');
function m = metrics(L, i_err, i_pla, i_ref)
  e = L(:, i_err);
  e = e((i_ref):end);
  pla = L(:, i_pla);
  m.mae = mean(abs(e));
  m.iae = sum(abs(e)) * CTRL_DT;
  m.overshoot = max(0, max(e));
  m.final_abs = abs(e(end));
  m.sum_dpla = sum(abs(diff(pla)));
  m.jitter = std(diff(pla));
  % 调节时间（首次进入 ±5rpm 并保持）
  m.settle = -1;
  for j = 1:numel(e)
    if all(abs(e(j:end)) < 5)
      m.settle = (j + i_ref - 2) * CTRL_DT;
      break;
    end
  end
end

M_mpc = metrics(log_mpc, 5, 4, i_ref+1);
M_pi = metrics(log_pi, 5, 4, i_ref+1);

fprintf(outfid,'  %-20s %14s %14s\n','指标','GRU-MPC','PI(精调)');
fprintf(outfid,'  %s\n', repmat('-',1,52));
fprintf(outfid,'  %-20s %14.2f %14.2f\n','MAE (rpm)', M_mpc.mae, M_pi.mae);
fprintf(outfid,'  %-20s %14.1f %14.1f\n','IAE (rpm·s)', M_mpc.iae, M_pi.iae);
fprintf(outfid,'  %-20s %14.2f %14.2f\n','超调 (rpm)', M_mpc.overshoot, M_pi.overshoot);
fprintf(outfid,'  %-20s %14.1f %14.1f\n','调节时间 (s)', M_mpc.settle, M_pi.settle);
fprintf(outfid,'  %-20s %14.2f %14.2f\n','稳态误差 (rpm)', M_mpc.final_abs, M_pi.final_abs);
fprintf(outfid,'  %-20s %14.3f %14.3f\n','Σ|dPLA| (deg)', M_mpc.sum_dpla, M_pi.sum_dpla);
fprintf(outfid,'  %-20s %14.4f %14.4f\n','PLA抖动 std', M_mpc.jitter, M_pi.jitter);

fprintf(outfid,'\n=== MPC 计算开销（%d 步）===\n', N_CL);
fprintf(outfid,'  中位=%.1f ms  P95=%.1f ms  最大=%.1f ms\n', ...
        median(log_mpc(:,6)), prctile(log_mpc(:,6),95), max(log_mpc(:,6)));
fprintf(outfid,'  控制周期=%.0f ms -> %s\n', CTRL_DT*1000, ...
        ternary(median(log_mpc(:,6)) < CTRL_DT*1000, 'MPC 满足预算', 'MPC 超预算'));
fprintf(outfid,'  Simulink 单步: 中位=%.0f ms（评估成本，不计入控制预算）\n', ...
        median(log_mpc(:,7)));

save(fullfile(outDir,'closedloop.mat'), 'log_mpc', 'log_pi', ...
     'n2_steady', 'ACT', 'Kp', 'Ki');
fclose(outfid);
fprintf('PHYS_OK\n');

function s = ternary(c,a,b)
if c, s=a; else, s=b; end
end


% 实验 B：CEM 前段权重消融（w_short=3.0 vs 1.0）
%
% 固定：相同历史状态、GRU 模型、随机种子、其余 CEM 参数
% 选择 k=10, 14, 17 三个关键状态
%
% 注意：两种权重对应不同目标函数，J 不可直接比较
% 必须用统一评估指标

outDir = 'C:\pclwm\pclwm_stage3a\causal';
addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');

fprintf('=== 实验 B：w_short 消融 ===\n\n');

% ══ 从 CSV 重建历史 ══
S = readtable(fullfile(outDir,'steps.csv'));
N = height(S);
WINDOW = 32; H = 32; DT = 0.105;
N2_REF = 5099.29;

OBS_ALL = [S.obs_wf, S.obs_n2, S.obs_t4, S.obs_t45, S.obs_p36];
ACT_ALL = [S.pla_apply, zeros(N,1), zeros(N,1)];
obs_steady = OBS_ALL(1,:);
act_steady = [55, 0, 0];

% ══ 基准 CFG（与 causal_run.m 完全一致）══
BASE = struct();
BASE.H = H; BASE.n_cand = 32; BASE.n_iter = 6; BASE.n_short = 8;
BASE.w_n2 = 1.0; BASE.w_du = 1.0; BASE.w_coast = 0.0;
BASE.pla_min = 48; BASE.pla_max = 78; BASE.dpla_max = 2.0;
BASE.warm_start = true;

KEY_K = [10, 14, 17];
W_SHORT_LIST = [3.0, 1.0];

RESULTS = struct();

for ki = 1:numel(KEY_K)
  k = KEY_K(ki);
  i = k - 1;

  % 历史窗口
  lo = i - WINDOW + 1;
  if lo < 0
    pad = -lo;
    obs_h = [repmat(obs_steady, pad, 1); OBS_ALL(1:i+1,:)];
    act_h = [repmat(act_steady, pad, 1); ACT_ALL(1:i+1,:)];
  else
    obs_h = OBS_ALL(lo:i+1, :);
    act_h = ACT_ALL(lo:i+1, :);
  end
  pla_prev = S.pla_prev(i+1);
  seed = k * 17;

  fprintf('── k=%d, t=%.2fs, PLA_prev=%.2f ──\n', k, S.t(i+1), pla_prev);

  ents = {};
  for wi = 1:numel(W_SHORT_LIST)
    ws = W_SHORT_LIST(wi);
    CFG = BASE;
    CFG.w_short = ws;
    CFG.mu_prev = [];       % 不用 warm-start（隔离变量）

    [pla_seq, info] = mpc_opt_diag(obs_h, act_h, N2_REF, pla_prev, CFG, P, seed);

    % ══ 统一评估指标（与 w_short 无关）══
    % 等权 MSE（不用任何加权）
    e = info.pred_n2_mu - N2_REF;
    mse_eq = mean(e.^2);
    mae_eq = mean(abs(e));
    peak = max(info.pred_n2_mu);
    ovs = peak - N2_REF;
    % 动作累计变化
    sum_dpla = sum(abs(diff([pla_prev, info.pla_seq])));
    % 末值误差
    e_term = info.pla_seq(end);  % 占位
    err_term = info.pred_n2_mu(end) - N2_REF;

    fprintf('  w_short=%.1f: PLA[1]=%.2f, 预测峰值=%.1f (+%.0f)\n', ...
            ws, info.pla_seq(1), peak, ovs);
    fprintf('             等权MSE=%.1f, 等权MAE=%.1f, 末值误差=%+.1f, Σ|dPLA|=%.2f\n', ...
            mse_eq, mae_eq, err_term, sum_dpla);
    fprintf('             PLA序列: [%.1f %.1f %.1f %.1f %.1f ... %.1f]\n', ...
            info.pla_seq(1), info.pla_seq(2), info.pla_seq(3), ...
            info.pla_seq(4), info.pla_seq(5), info.pla_seq(end));

    ents{wi} = struct('w_short', ws, 'pla_seq', info.pla_seq, ...
      'pred_n2', info.pred_n2_mu, 'peak', peak, 'overshoot', ovs, ...
      'mse_eq', mse_eq, 'mae_eq', mae_eq, 'err_term', err_term, ...
      'sum_dpla', sum_dpla, 'j_own', info.J_final, ...
      'pred_all', info.pred_all_phys);
  end

  % ══ 配对对比 ══
  a0 = ents{1}; a1 = ents{2};
  fprintf('\n  【配对对比】\n');
  fprintf('    %-22s %14s %14s %12s\n','指标','w_short=3.0','w_short=1.0','变化');
  fprintf('    %s\n', repmat('-',1,66));
  fprintf('    %-22s %14.2f %14.2f %+12.2f\n','首步 PLA', ...
          a0.pla_seq(1), a1.pla_seq(1), a1.pla_seq(1)-a0.pla_seq(1));
  fprintf('    %-22s %14.1f %14.1f %+12.1f\n','预测峰值', ...
          a0.peak, a1.peak, a1.peak-a0.peak);
  fprintf('    %-22s %14.1f %14.1f %+12.1f\n','预测超调', ...
          a0.overshoot, a1.overshoot, a1.overshoot-a0.overshoot);
  fprintf('    %-22s %14.1f %14.1f %+12.1f\n','等权MSE', ...
          a0.mse_eq, a1.mse_eq, a1.mse_eq-a0.mse_eq);
  fprintf('    %-22s %14.1f %14.1f %+12.1f\n','等权MAE', ...
          a0.mae_eq, a1.mae_eq, a1.mae_eq-a0.mae_eq);
  fprintf('    %-22s %14.1f %14.1f %+12.1f\n','末值误差', ...
          a0.err_term, a1.err_term, a1.err_term-a0.err_term);
  fprintf('    %-22s %14.2f %14.2f %+12.2f\n','Σ|dPLA|', ...
          a0.sum_dpla, a1.sum_dpla, a1.sum_dpla-a0.sum_dpla);
  fprintf('\n');

  RESULTS(ki).k = k;
  RESULTS(ki).t = S.t(i+1);
  RESULTS(ki).ent = ents;
end

% ══ 保存 ══
save(fullfile(outDir,'expB_ablation.mat'), 'RESULTS');
fprintf('  -> expB_ablation.mat\n');
fprintf('PHYS_OK\n');

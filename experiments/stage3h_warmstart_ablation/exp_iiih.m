% III-H 三配置离线消融
%
% 问题：warm-start 依赖"上一轮的 mu"。
%       causal 日志只保存了 pla_seq（约束后），不是 CEM 的原始 mu。
%       但 mpc_opt_diag 的 info.mu 保存了原始 mu！
%
% 方案：
%   从 causal_diag.mat 读取每步的 info.mu，重建 warm-start 链路
%   然后在 k=10/14/17 三个状态上跑三配置

outDir = 'C:\pclwm\pclwm_stage3a\iiih';
if ~isfolder(outDir), mkdir(outDir); end
addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');

% ══ 读 causal 数据 ══
S = readtable('C:\pclwm\pclwm_stage3a\causal\steps.csv');
N = height(S);
WINDOW = 32; H = 32; DT = 0.105; N2_REF = 5099.29;

OBS_ALL = [S.obs_wf, S.obs_n2, S.obs_t4, S.obs_t45, S.obs_p36];
ACT_ALL = [S.pla_apply, zeros(N,1), zeros(N,1)];
obs_steady = OBS_ALL(1,:);
act_steady = [55, 0, 0];

% ══ 读原始 mu（从 causal_diag.mat）══
fprintf('=== 读取原始 mu ===\n');
try
  CD = load('C:\pclwm\pclwm_stage3a\causal\causal_diag.mat', 'DIAG');
  DIAG = CD.DIAG;
  has_mu = true;
  fprintf('  ✅ causal_diag.mat 已加载\n');
catch
  has_mu = false;
  fprintf('  ⚠ 无法加载 causal_diag.mat\n');
end

% ══ 基准 CFG ══
BASE = struct();
BASE.H = H; BASE.n_cand = 32; BASE.n_iter = 6; BASE.n_short = 8;
BASE.w_n2 = 1.0; BASE.w_du = 1.0;
BASE.pla_min = 48; BASE.pla_max = 78; BASE.dpla_max = 2.0;
BASE.warm_start = true; BASE.mu_prev = []; BASE.ws_shift = false;

KEY_K = [10, 14, 17];

fprintf('\n=== 三配置消融 ===\n');
fprintf('  A: 原始 warm-start, w_short=3\n');
fprintf('  B: 移位 warm-start, w_short=3\n');
fprintf('  C: 移位 warm-start, w_short=1\n\n');

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

  % ★ 上一轮的原始 mu
  mu_prev = [];
  if has_mu
    % 检查 cell 索引对不对
    kk = k;  % DIAG.steps{k} 对应第 k 步
    assert(DIAG.steps{kk}.k == k, 'DIAG.steps 索引不匹配');
    % 用 pla_seq_full = 约束后的最终可执行动作序列（任务要求）
    mu_prev = DIAG.steps{kk}.pla_seq_full;
  end
  if isempty(mu_prev)
    mu_prev = S.pla_apply(i+1) * ones(1,H);
  end
  assert(numel(mu_prev) == H, 'mu 长度不符');

  fprintf('── k=%d, t=%.2fs, PLA_prev=%.2f ──\n', k, S.t(i+1), pla_prev);
  fprintf('  上一轮 mu（前5）: [%.2f %.2f %.2f %.2f %.2f]\n', mu_prev(1:5));

  % ══ 三配置 ══
  CFGS = {};
  CFGS{1} = BASE; CFGS{1}.w_short = 3.0; CFGS{1}.ws_shift = false;
  CFGS{2} = BASE; CFGS{2}.w_short = 3.0; CFGS{2}.ws_shift = true;
  CFGS{3} = BASE; CFGS{3}.w_short = 1.0; CFGS{3}.ws_shift = true;
  NAMES = {'A:原始ws,w=3', 'B:移位ws,w=3', 'C:移位ws,w=1'};

  ents = {};
  for ci = 1:3
    CFG = CFGS{ci};
    CFG.mu_prev = mu_prev;
    [pla, info] = mpc_opt_ws(obs_h, act_h, N2_REF, pla_prev, CFG, P, seed);

    % 统一评价：等权 MSE（不依赖 w_short）
    e = info.pred_n2 - N2_REF;
    mse_eq = mean(e.^2);
    mae_eq = mean(abs(e));
    sum_dpla = sum(abs(diff([pla_prev, pla])));
    err_term = info.pred_n2(end) - N2_REF;

    fprintf('  %-16s PLA[1]=%.2f  peak=%.1f (+%.0f)  MSE_eq=%.1f  MAE=%.1f  SdPLA=%.1f  viol=%d/%d\n', ...
            NAMES{ci}, pla(1), info.pred_peak, info.pred_overshoot, ...
            mse_eq, mae_eq, sum_dpla, info.viol_bound, info.viol_rate);

    ents{ci} = struct('name', NAMES{ci}, 'pla', pla, 'pred_n2', info.pred_n2, ...
      'peak', info.pred_peak, 'overshoot', info.pred_overshoot, ...
      'peak_k', info.pred_peak_k, 'mse_eq', mse_eq, 'mae_eq', mae_eq, ...
      'sum_dpla', sum_dpla, 'err_term', err_term, ...
      'viol_bound', info.viol_bound, 'viol_rate', info.viol_rate, ...
      'iter_log', info.iter_log);
  end

  % ══ 配对对比 ══
  fprintf('\n  【配对对比】\n');
  fprintf('    %-20s %12s %12s %12s\n','指标', NAMES{:});
  fprintf('    %s\n', repmat('-',1,60));
  labs = {'首步PLA','预测峰值','预测超调','等权MSE','等权MAE','Σ|dPLA|'};
  vals = [ents{1}.pla(1), ents{2}.pla(1), ents{3}.pla(1); ...
          ents{1}.peak, ents{2}.peak, ents{3}.peak; ...
          ents{1}.overshoot, ents{2}.overshoot, ents{3}.overshoot; ...
          ents{1}.mse_eq, ents{2}.mse_eq, ents{3}.mse_eq; ...
          ents{1}.mae_eq, ents{2}.mae_eq, ents{3}.mae_eq; ...
          ents{1}.sum_dpla, ents{2}.sum_dpla, ents{3}.sum_dpla];
  for r=1:numel(labs)
    fprintf('    %-20s %12.2f %12.2f %12.2f\n', labs{r}, vals(r,:));
  end
  fprintf('\n');

  RESULTS(ki).k = k;
  RESULTS(ki).t = S.t(i+1);
  RESULTS(ki).pla_prev = pla_prev;
  RESULTS(ki).mu_prev = mu_prev;
  RESULTS(ki).ent = ents;
end

% ══ 保存为 CSV ══
fid = fopen(fullfile(outDir,'ablation3.csv'),'w');
fprintf(fid,'k,t,cfg,pla1,pla_seq,peak,overshoot,peak_k,mse_eq,mae_eq,sum_dpla,err_term,viol_bound,viol_rate\n');
for ki=1:numel(RESULTS)
  for ci=1:3
    e = RESULTS(ki).ent{ci};
    fprintf(fid,'%d,%.4f,%s,%.4f,"%s",%.4f,%.4f,%d,%.2f,%.2f,%.4f,%.4f,%d,%d\n', ...
      RESULTS(ki).k, RESULTS(ki).t, NAMES{ci}, e.pla(1), ...
      mat2str(round(e.pla,3)), e.peak, e.overshoot, e.peak_k, ...
      e.mse_eq, e.mae_eq, e.sum_dpla, e.err_term, e.viol_bound, e.viol_rate);
  end
end
fclose(fid);
fprintf('  -> ablation3.csv\n');

% ══ 保存完整序列（供绘图）══
fid = fopen(fullfile(outDir,'pla_seq3.csv'),'w');
fprintf(fid,'k,cfg,step,pla,pred_n2\n');
for ki=1:numel(RESULTS)
  for ci=1:3
    e = RESULTS(ki).ent{ci};
    for j=1:H
      fprintf(fid,'%d,%s,%d,%.4f,%.4f\n', RESULTS(ki).k, NAMES{ci}, j, ...
        e.pla(j), e.pred_n2(j));
    end
  end
end
fclose(fid);
fprintf('  -> pla_seq3.csv\n');

save(fullfile(outDir,'ablation3.mat'), 'RESULTS');
fprintf('  -> ablation3.mat\n');
fprintf('PHYS_OK\n');




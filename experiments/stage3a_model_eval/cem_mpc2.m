function [pla_best, info] = cem_mpc2(obs_hist, act_hist, n2_ref, pla_prev, CFG, P, seed)
%CEM_MPC2 改进版
%
% 修复 v1 的问题：
%   v1 代价 = 整段 H 步平均，导致前几步的权重被稀释
%   v2 代价 = 前 n_short 步重点加权 + 整段平滑项
%
% 另外：用 log-barrier 惩罚接近边界的解

H = CFG.H;
rng(seed);
mu = pla_prev * ones(1, H);
sig = 0.5 * ones(1, H);
obs_z = (obs_hist - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
act_z = (act_hist - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);

n_short = CFG.n_short;      % 前 n 步重点加权
w_short = 3.0;              % 前段权重倍数

% 权重向量：前 n_short 步权重更高
wv = ones(H, 1);
wv(1:n_short) = w_short;
wv = wv / sum(wv);

for it = 1:CFG.n_iter
  cand = mu + sig .* randn(CFG.n_cand, H);
  cand = min(CFG.pla_max, max(CFG.pla_min, cand));
  for j = 2:H
    cand(:,j) = min(cand(:,j-1)+CFG.dpla_max, ...
                    max(cand(:,j-1)-CFG.dpla_max, cand(:,j)));
  end

  J = zeros(CFG.n_cand, 1);
  for c = 1:CFG.n_cand
    af_phys = [cand(c,:)', zeros(H,2)];
    af_z = (af_phys - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
    pred_z = gru_forward(obs_z, act_z, af_z, P);
    pred_n2 = pred_z(:,2) * P.obs_sd(2) + P.obs_mu(2);
    % 加权跟踪误差
    j_track = sum(wv .* (pred_n2 - n2_ref).^2);
    % 平滑项
    j_du = mean(diff([pla_prev, cand(c,:)]).^2);
    % 边界惩罚（接近 50/60 时加罚）
    margin = min(cand(c,:) - CFG.pla_min, CFG.pla_max - cand(c,:));
    j_bnd = 10.0 * sum(max(0, 0.5 - margin).^2);
    J(c) = CFG.w_n2*j_track + CFG.w_du*j_du + j_bnd;
  end

  [~, idx] = sort(J);
  elite = cand(idx(1:CFG.n_elite), :);
  mu  = mean(elite, 1);
  sig = std(elite, 0, 1) + 1e-3;
end
pla_best = min(CFG.pla_max, max(CFG.pla_min, mu));
info = struct('n_cand', CFG.n_cand, 'mu1', mu(1));
end

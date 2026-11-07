function profile_cem()
% 任务2：CEM 计算效率剖析
%
% 分解各部分耗时：
%   1. 历史状态编码（GRU 序列 + LayerNorm）
%   2. 候选任务构造（数据转换）
%   3. GRU 候选预测（批量）
%   4. CEM 精英更新
%
% 并测试候选数/迭代数的速度-精度关系

addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');

outDir = 'C:\pclwm\pclwm_stage3a\iiic';
if ~isfolder(outDir), mkdir(outDir); end
outfid = fopen(fullfile(outDir,'cem_profile.txt'),'w','n','UTF-8');

fprintf(outfid,'=== CEM 计算效率剖析 ===\n\n');

% 稳态观测
obs_steady = [0.6180, 4349.29, 2465.17, 1715.78, 303.61];
obs_hist = repmat(obs_steady, 32, 1);
act_hist = repmat([55,0,0], 32, 1);
obs_z = (obs_hist - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
act_z = (act_hist - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
H = 32;

% ══════════════════════════════════════════════════
% [1] 各部分耗时
% ══════════════════════════════════════════════════
fprintf(outfid,'[1] 单次调用耗时分解（N=50 次平均）\n');

N_REP = 50;
t_enc = zeros(N_REP,1);
t_conv = zeros(N_REP,1);
t_pred1 = zeros(N_REP,1);
t_pred32 = zeros(N_REP,1);
t_elite = zeros(N_REP,1);

n_cand = 32;

for r = 1:N_REP
  % (a) 历史编码
  tic;
  [h_last, h3] = encode_history(obs_z, act_z, P);
  t_enc(r) = toc;

  % (b) 候选构造
  tic;
  cand = 55 + 0.5*randn(n_cand, H);
  cand = min(60, max(50, cand));
  af_all = zeros(n_cand, H, 3);
  for c = 1:n_cand
    af_all(c,:,1) = (cand(c,:) - P.act_mu(1)) / P.act_sd(1);
    af_all(c,:,2) = (0 - P.act_mu(2)) / P.act_sd(2);
    af_all(c,:,3) = (0 - P.act_mu(3)) / P.act_sd(3);
  end
  t_conv(r) = toc;

  % (c) 单候选预测
  tic;
  p1 = gru_rollout_one(h_last, obs_z(end,:), squeeze(af_all(1,:,:)), P);
  t_pred1(r) = toc;

  % (d) 32 候选批量预测（关键：向量化）
  tic;
  P32 = gru_rollout_batch(h_last, obs_z(end,:), af_all, P);
  t_pred32(r) = toc;

  % (e) 精英更新
  tic;
  J = sum((P32(:,:,2)*P.obs_sd(2)+P.obs_mu(2) - 4400).^2, 2);
  [~, idx] = sort(J);
  el = cand(idx(1:8), :);
  mu_n = mean(el,1); sig_n = std(el,0,1);
  t_elite(r) = toc;
end

fprintf(outfid,'  %-28s %10s %10s %10s\n','环节','中位(ms)','P95(ms)','最大(ms)');
fprintf(outfid,'  %-28s %10.3f %10.3f %10.3f\n','历史编码', ...
        median(t_enc)*1000, prctile(t_enc,95)*1000, max(t_enc)*1000);
fprintf(outfid,'  %-28s %10.3f %10.3f %10.3f\n','候选构造', ...
        median(t_conv)*1000, prctile(t_conv,95)*1000, max(t_conv)*1000);
fprintf(outfid,'  %-28s %10.3f %10.3f %10.3f\n','单候选预测', ...
        median(t_pred1)*1000, prctile(t_pred1,95)*1000, max(t_pred1)*1000);
fprintf(outfid,'  %-28s %10.3f %10.3f %10.3f\n','32候选批量预测', ...
        median(t_pred32)*1000, prctile(t_pred32,95)*1000, max(t_pred32)*1000);
fprintf(outfid,'  %-28s %10.3f %10.3f %10.3f\n','精英更新', ...
        median(t_elite)*1000, prctile(t_elite,95)*1000, max(t_elite)*1000);

fprintf(outfid,'\n  串行 vs 批量（32 候选）：\n');
fprintf(outfid,'    串行 32 x %.3f ms = %.1f ms\n', ...
        median(t_pred1)*1000, 32*median(t_pred1)*1000);
fprintf(outfid,'    批量          = %.1f ms\n', median(t_pred32)*1000);
fprintf(outfid,'    加速比 = %.1fx\n', 32*median(t_pred1)/median(t_pred32));

% ══════════════════════════════════════════════════
% [2] 候选数 / 迭代数 扫描
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[2] 端到端 MPC 耗时 vs 配置\n');
fprintf(outfid,'  %8s %8s %12s %12s %12s %14s\n', ...
        'n_cand','n_iter','中位(ms)','P95(ms)','最大(ms)','跟踪J');
fprintf(outfid,'  %s\n', repmat('-',1,74));

n2_ref = 4399.29;
best_cfg = [];
for nc = [4 8 16 32 64]
  for ni = [1 2 4 6]
    ts = zeros(20,1); Js = zeros(20,1);
    for r = 1:20
      tic;
      [pla_seq, J] = cem_fast(obs_z, act_z, n2_ref, 55.0, nc, ni, H, P, r*13);
      ts(r) = toc;
      Js(r) = J;
    end
    fprintf(outfid,'  %8d %8d %12.2f %12.2f %12.2f %14.1f\n', ...
            nc, ni, median(ts)*1000, prctile(ts,95)*1000, max(ts)*1000, median(Js));
    best_cfg(end+1,:) = [nc, ni, median(ts)*1000, median(Js)];
  end
end

fprintf(outfid,'\n[3] 满足 105ms 的配置\n');
ok = best_cfg(best_cfg(:,3) < 105, :);
if isempty(ok)
  fprintf(outfid,'  无！\n');
else
  fprintf(outfid,'  %8s %8s %12s %14s\n','n_cand','n_iter','中位(ms)','跟踪J');
  for i=1:size(ok,1)
    fprintf(outfid,'  %8d %8d %12.2f %14.1f\n', ok(i,1), ok(i,2), ok(i,3), ok(i,4));
  end
  [~,bi] = min(ok(:,4));
  fprintf(outfid,'\n  最低 J 的可行配置: n_cand=%d, n_iter=%d (%.1f ms, J=%.1f)\n', ...
          ok(bi,1), ok(bi,2), ok(bi,3), ok(bi,4));
end

fclose(outfid);
fprintf('PHYS_OK\n');
end

% ══════════════════════════════════════════════════
function [h_last, h3] = encode_history(obs_z, act_z, P)
W = size(obs_z,1);
x = [obs_z, act_z];
h1 = x * P.w_inp_0_weight' + reshape(P.w_inp_0_bias,1,[]);
mu = mean(h1,2); va = mean((h1-mu).^2,2);
h2 = (h1-mu)./sqrt(va+1e-5).*reshape(P.w_inp_1_weight,1,[]) + reshape(P.w_inp_1_bias,1,[]);
h3 = h2 .* (1./(1+exp(-h2)));
hh = zeros(1,128);
for t = 1:W
  hh = gru_step(h3(t,:), hh, P.w_gru_weight_ih_l0, P.w_gru_weight_hh_l0, ...
                P.w_gru_bias_ih_l0, P.w_gru_bias_hh_l0);
end
h_last = hh;
end

function h = gru_step(x, h, wih, whh, bih, bhh)
x = reshape(x,1,[]); h = reshape(h,1,[]);
bih = reshape(bih,1,[]); bhh = reshape(bhh,1,[]);
Hg = size(whh,1)/3;
xr = x*wih(1:Hg,:)' + bih(1:Hg);
xz = x*wih(Hg+1:2*Hg,:)' + bih(Hg+1:2*Hg);
xn = x*wih(2*Hg+1:3*Hg,:)' + bih(2*Hg+1:3*Hg);
hr = h*whh(1:Hg,:)' + bhh(1:Hg);
hz = h*whh(Hg+1:2*Hg,:)' + bhh(Hg+1:2*Hg);
hn = h*whh(2*Hg+1:3*Hg,:)' + bhh(2*Hg+1:3*Hg);
r = 1./(1+exp(-(xr+hr))); z = 1./(1+exp(-(xz+hz)));
nn = tanh(xn + r.*hn);
h = (1-z).*nn + z.*h;
end

function pred = gru_rollout_one(h, o_prev, af, P)
H = size(af,1);
pred = zeros(H,5);
for k = 1:H
  z = [o_prev, af(k,:)] * P.w_o2h_weight' + reshape(P.w_o2h_bias,1,[]);
  h = gru_step(z, h, P.w_cell_weight_ih, P.w_cell_weight_hh, ...
               P.w_cell_bias_ih, P.w_cell_bias_hh);
  a1 = h*P.w_dec_0_weight' + reshape(P.w_dec_0_bias,1,[]);
  a1 = a1.*(1./(1+exp(-a1)));
  a2 = a1*P.w_dec_2_weight' + reshape(P.w_dec_2_bias,1,[]);
  a2 = a2.*(1./(1+exp(-a2)));
  o = a2*P.w_dec_4_weight' + reshape(P.w_dec_4_bias,1,[]);
  pred(k,:) = o;
  o_prev = o;
end
end

function pred = gru_rollout_batch(h0, o_prev0, af_all, P)
% 批量递推：af_all 是 (B,H,3)
B = size(af_all,1); H = size(af_all,2);
h = repmat(h0, B, 1);          % (B,128)
o_prev = repmat(o_prev0, B, 1); % (B,5)
pred = zeros(B,H,5);
for k = 1:H
  afk = reshape(af_all(:,k,:), B, 3);
  z = [o_prev, afk] * P.w_o2h_weight' ...
      + repmat(reshape(P.w_o2h_bias,1,[]), B, 1);
  h = gru_step_batch(z, h, P.w_cell_weight_ih, P.w_cell_weight_hh, ...
                     P.w_cell_bias_ih, P.w_cell_bias_hh);
  a1 = h*P.w_dec_0_weight' + repmat(reshape(P.w_dec_0_bias,1,[]), B, 1);
  a1 = a1.*(1./(1+exp(-a1)));
  a2 = a1*P.w_dec_2_weight' + repmat(reshape(P.w_dec_2_bias,1,[]), B, 1);
  a2 = a2.*(1./(1+exp(-a2)));
  o = a2*P.w_dec_4_weight' + repmat(reshape(P.w_dec_4_bias,1,[]), B, 1);
  pred(:,k,:) = o;
  o_prev = o;
end
end

function h = gru_step_batch(x, h, wih, whh, bih, bhh)
% x:(B,in), h:(B,H)
Hg = size(whh,1)/3;
xr = x*wih(1:Hg,:)' + reshape(bih(1:Hg),1,[]);
xz = x*wih(Hg+1:2*Hg,:)' + reshape(bih(Hg+1:2*Hg),1,[]);
xn = x*wih(2*Hg+1:3*Hg,:)' + reshape(bih(2*Hg+1:3*Hg),1,[]);
hr = h*whh(1:Hg,:)' + reshape(bhh(1:Hg),1,[]);
hz = h*whh(Hg+1:2*Hg,:)' + reshape(bhh(Hg+1:2*Hg),1,[]);
hn = h*whh(2*Hg+1:3*Hg,:)' + reshape(bhh(2*Hg+1:3*Hg),1,[]);
r = 1./(1+exp(-(xr+hr))); z = 1./(1+exp(-(xz+hz)));
nn = tanh(xn + r.*hn);
h = (1-z).*nn + z.*h;
end

function [pla, Jbest] = cem_fast(obs_z, act_z, n2_ref, pla0, nc, ni, H, P, seed)
% 快速版 CEM（批量 + 复用历史编码）
rng(seed);
[h0, ~] = encode_history(obs_z, act_z, P);
o_prev0 = obs_z(end,:);
mu = pla0*ones(1,H); sig = 0.5*ones(1,H);
n_elite = max(2, floor(nc/4));
Jbest = inf;
for it = 1:ni
  cand = mu + sig .* randn(nc, H);
  cand = min(60, max(50, cand));
  af_all = zeros(nc,H,3);
  af_all(:,:,1) = (cand - P.act_mu(1))/P.act_sd(1);
  af_all(:,:,2) = (0 - P.act_mu(2))/P.act_sd(2);
  af_all(:,:,3) = (0 - P.act_mu(3))/P.act_sd(3);
  pred = gru_rollout_batch(h0, o_prev0, af_all, P);
  n2p = pred(:,:,2)*P.obs_sd(2) + P.obs_mu(2);
  J = sum((n2p - n2_ref).^2, 2);
  [Js, idx] = sort(J);
  if Js(1) < Jbest, Jbest = Js(1); end
  el = cand(idx(1:n_elite),:);
  mu = mean(el,1); sig = std(el,0,1) + 1e-3;
end
pla = min(60, max(50, mu));
end


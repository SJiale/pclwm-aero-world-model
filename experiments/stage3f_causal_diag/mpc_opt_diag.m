function [pla_seq, info] = mpc_opt_diag(obs_hist, act_hist, n2_ref, pla_prev, CFG, P, seed)
%MPC_OPT_DIAG 与 mpc_opt 完全相同的算法，但输出完整诊断
%
% 不修改任何算法逻辑、参数、权重
% 仅额外返回：
%   info.mu            最终最优动作序列 (1,H) 物理 PLA
%   info.pred_n2_mu    用 mu 预测的未来 N2 (1,H) 物理 rpm
%   info.pred_peak_n2  mu 预测的峰值 N2
%   info.pred_peak_k   mu 预测的峰值步
%   info.j_track_final mu 的跟踪代价
%   info.j_du_final    mu 的动作代价
%   info.J_final       mu 的总代价
%   info.iter_log      每轮 CEM 的最优代价 + 对应预测

H = CFG.H;
t0 = tic;

% ══ 1. 历史编码 ══
obs_z = (obs_hist - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
act_z = (act_hist - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
[h0, ~] = enc_hist(obs_z, act_z, P);
o0 = obs_z(end, :);
t_enc = toc(t0);

% ══ 2. CEM 初始化 ══
rng(seed);
if isfield(CFG,'warm_start') && CFG.warm_start && ~isempty(CFG.mu_prev)
  mu = CFG.mu_prev(:)';
else
  mu = pla_prev * ones(1, H);
end
sig = 0.5 * ones(1, H);
n_el = max(2, floor(CFG.n_cand/4));

wv = ones(1,H);
wv(1:CFG.n_short) = CFG.w_short;
wv = wv / sum(wv);

% ══ 3. CEM 迭代（完全同 mpc_opt）══
iter_log = struct('it', {}, 'J_best', {}, 'mu', {}, 'pred_n2', {});
t_cem = 0;
for it = 1:CFG.n_iter
  tc0 = tic;
  cand = mu + sig .* randn(CFG.n_cand, H);
  cand = min(CFG.pla_max, max(CFG.pla_min, cand));
  cand = apply_rate_limit(cand, pla_prev, CFG.dpla_max);

  af = zeros(CFG.n_cand, H, 3);
  af(:,:,1) = (cand - P.act_mu(1)) / P.act_sd(1);
  af(:,:,2) = (0 - P.act_mu(2)) / P.act_sd(2);
  af(:,:,3) = (0 - P.act_mu(3)) / P.act_sd(3);
  pr = rollout_batch(h0, o0, af, P);
  n2p = pr(:,:,2) * P.obs_sd(2) + P.obs_mu(2);

  j_track = sum(wv .* (n2p - n2_ref).^2, 2);
  du = [pla_prev*ones(CFG.n_cand,1), cand];
  j_du = sum(diff(du,1,2).^2, 2);
  j_coast = sum(cand.^2, 2) * CFG.w_coast;

  J = CFG.w_n2*j_track + CFG.w_du*j_du + j_coast;

  [Js, idx] = sort(J);
  el = cand(idx(1:n_el), :);

  % ★ 记录本轮：最优候选的代价与预测
  b = idx(1);
  iter_log(it).it = it;
  iter_log(it).J_best = Js(1);
  iter_log(it).mu = cand(b,:);
  iter_log(it).pred_n2 = n2p(b,:);
  iter_log(it).j_track = j_track(b);
  iter_log(it).j_du = j_du(b);

  mu = mean(el, 1);
  sig = std(el, 0, 1) + 1e-3;
  t_cem = t_cem + toc(tc0);
end

% ══ 4. 最终解 + 对最终 mu 的预测 ══
pla_seq = min(CFG.pla_max, max(CFG.pla_min, mu));
pla_seq = apply_rate_limit(pla_seq, pla_prev, CFG.dpla_max);

% ★ 用最终动作序列重新预测（关键：这是"MPC 认为会发生什么"）
af_mu = zeros(1, H, 3);
af_mu(1,:,1) = (pla_seq - P.act_mu(1)) / P.act_sd(1);
af_mu(1,:,2) = (0 - P.act_mu(2)) / P.act_sd(2);
af_mu(1,:,3) = (0 - P.act_mu(3)) / P.act_sd(3);
pr_mu = rollout_batch(h0, o0, af_mu, P);
pred_all = reshape(squeeze(pr_mu(1,:,:)), H, 5);   % (H,5) 归一化
pred_all_phys = pred_all .* reshape(P.obs_sd,1,[]) + reshape(P.obs_mu,1,[]);
pred_n2_mu = pred_all_phys(:,2)';            % (1,H) rpm

[pk_n2, pk_k] = max(pred_n2_mu);

% 最终 mu 的代价
j_track_f = sum(wv .* (pred_n2_mu - n2_ref).^2);
du_f = [pla_prev, pla_seq];
j_du_f = sum(diff(du_f).^2);
j_coast_f = sum(pla_seq.^2) * CFG.w_coast;
J_f = CFG.w_n2*j_track_f + CFG.w_du*j_du_f + j_coast_f;

info = struct();
info.t_total_ms = toc(t0)*1000;
info.t_enc_ms = t_enc*1000;
info.t_cem_ms = t_cem*1000;
info.mu = mu;
info.pla_seq = pla_seq;
info.pred_n2_mu = pred_n2_mu;
info.pred_all_phys = pred_all_phys;
info.pred_peak_n2 = pk_n2;
info.pred_peak_k = pk_k;
info.j_track_final = j_track_f;
info.j_du_final = j_du_f;
info.j_coast_final = j_coast_f;
info.J_final = J_f;
info.iter_log = iter_log;
info.wv = wv;
info.n2_ref = n2_ref;
info.pla_prev = pla_prev;
end

% ══════════════════════════════════════════════════
function c = apply_rate_limit(c, pla_prev, dmax)
n = size(c,1);
c(:,1) = min(pla_prev+dmax, max(pla_prev-dmax, c(:,1)));
for j = 2:size(c,2)
  c(:,j) = min(c(:,j-1)+dmax, max(c(:,j-1)-dmax, c(:,j)));
end
end

function [h_last, h3] = enc_hist(obs_z, act_z, P)
W = size(obs_z,1);
x = [obs_z, act_z];
h1 = x*P.w_inp_0_weight' + reshape(P.w_inp_0_bias,1,[]);
mu = mean(h1,2); va = mean((h1-mu).^2,2);
h2 = (h1-mu)./sqrt(va+1e-5).*reshape(P.w_inp_1_weight,1,[]) + reshape(P.w_inp_1_bias,1,[]);
h3 = h2.*(1./(1+exp(-h2)));
hh = zeros(1,128);
for t=1:W
  hh = gru_step(h3(t,:), hh, P.w_gru_weight_ih_l0, P.w_gru_weight_hh_l0, ...
                P.w_gru_bias_ih_l0, P.w_gru_bias_hh_l0);
end
h_last = hh;
end

function h = gru_step(x,h,wih,whh,bih,bhh)
x=reshape(x,1,[]); h=reshape(h,1,[]);
bih=reshape(bih,1,[]); bhh=reshape(bhh,1,[]);
Hg=size(whh,1)/3;
xr=x*wih(1:Hg,:)'+bih(1:Hg); xz=x*wih(Hg+1:2*Hg,:)'+bih(Hg+1:2*Hg);
xn=x*wih(2*Hg+1:3*Hg,:)'+bih(2*Hg+1:3*Hg);
hr=h*whh(1:Hg,:)'+bhh(1:Hg); hz=h*whh(Hg+1:2*Hg,:)'+bhh(Hg+1:2*Hg);
hn=h*whh(2*Hg+1:3*Hg,:)'+bhh(2*Hg+1:3*Hg);
r=1./(1+exp(-(xr+hr))); z=1./(1+exp(-(xz+hz)));
nn=tanh(xn+r.*hn); h=(1-z).*nn+z.*h;
end

function pred = rollout_batch(h0, o_prev0, af_all, P)
B=size(af_all,1); H=size(af_all,2);
h=repmat(h0,B,1); o_prev=repmat(o_prev0,B,1); pred=zeros(B,H,5);
for k=1:H
  afk=reshape(af_all(:,k,:),B,3);
  z=[o_prev,afk]*P.w_o2h_weight' + repmat(reshape(P.w_o2h_bias,1,[]),B,1);
  h=gru_step_b(z,h,P.w_cell_weight_ih,P.w_cell_weight_hh,P.w_cell_bias_ih,P.w_cell_bias_hh);
  a1=h*P.w_dec_0_weight'+repmat(reshape(P.w_dec_0_bias,1,[]),B,1); a1=a1.*(1./(1+exp(-a1)));
  a2=a1*P.w_dec_2_weight'+repmat(reshape(P.w_dec_2_bias,1,[]),B,1); a2=a2.*(1./(1+exp(-a2)));
  o=a2*P.w_dec_4_weight'+repmat(reshape(P.w_dec_4_bias,1,[]),B,1);
  pred(:,k,:)=o; o_prev=o;
end
end

function h = gru_step_b(x,h,wih,whh,bih,bhh)
Hg=size(whh,1)/3;
xr=x*wih(1:Hg,:)'+reshape(bih(1:Hg),1,[]); xz=x*wih(Hg+1:2*Hg,:)'+reshape(bih(Hg+1:2*Hg),1,[]);
xn=x*wih(2*Hg+1:3*Hg,:)'+reshape(bih(2*Hg+1:3*Hg),1,[]);
hr=h*whh(1:Hg,:)'+reshape(bhh(1:Hg),1,[]); hz=h*whh(Hg+1:2*Hg,:)'+reshape(bhh(Hg+1:2*Hg),1,[]);
hn=h*whh(2*Hg+1:3*Hg,:)'+reshape(bhh(2*Hg+1:3*Hg),1,[]);
r=1./(1+exp(-(xr+hr))); z=1./(1+exp(-(xz+hz)));
nn=tanh(xn+r.*hn); h=(1-z).*nn+z.*h;
end


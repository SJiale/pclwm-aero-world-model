% 实验 B 补充：CEM 是否找到全局最优？
%
% 疑点：
%   k=17 时 CEM 选了 PLA≈70.8，预测峰值 5566（超调 +467）
%   这个代价显然很大，为什么 CEM 仍选它？
%
% 验证：
%   用同一 GRU、同一历史，评估几个【参考策略】的等权代价
%   若参考策略显著优于 CEM 解，说明 CEM 探索不足

outDir = 'C:\pclwm\pclwm_stage3a\causal';
addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');

S = readtable(fullfile(outDir,'steps.csv'));
N = height(S);
WINDOW = 32; H = 32; DT = 0.105;
N2_REF = 5099.29;

OBS_ALL = [S.obs_wf, S.obs_n2, S.obs_t4, S.obs_t45, S.obs_p36];
ACT_ALL = [S.pla_apply, zeros(N,1), zeros(N,1)];
obs_steady = OBS_ALL(1,:);
act_steady = [55, 0, 0];

fprintf('=== 实验 B 补充：CEM 是否找到全局最优 ===\n\n');

KEY_K = [10, 14, 17];
PLA_GOLD = 63.64;   % 对应 N2=5099 的稳态 PLA

for k = KEY_K
  i = k - 1;
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

  % ══ 参考策略 ══
  STRATS = {};
  % S1: 立即降到 GOLD 并保持
  STRATS{end+1} = struct('name','立即降至63.64', ...
    'pla', [PLA_GOLD*ones(1,H)]);
  % S2: 线性从 prev 降到 GOLD（前 10 步），之后保持
  ramp = zeros(1,H);
  for j=1:H
    if j <= 10
      ramp(j) = pla_prev + (PLA_GOLD-pla_prev)*j/10;
    else
      ramp(j) = PLA_GOLD;
    end
  end
  STRATS{end+1} = struct('name','10步线性降至63.64', 'pla', ramp);
  % S3: 保持 pla_prev
  STRATS{end+1} = struct('name','保持当前值', 'pla', pla_prev*ones(1,H));
  % S4: 前 5 步缓升 2°，之后降到 GOLD
  s4 = zeros(1,H);
  for j=1:H
    if j <= 5
      s4(j) = pla_prev + 0.4*j;
    else
      s4(j) = max(PLA_GOLD, s4(5) - (s4(5)-PLA_GOLD)*(j-5)/10);
    end
  end
  STRATS{end+1} = struct('name','前5步缓升后降', 'pla', s4);

  % ══ 评估 ══
  fprintf('── k=%d, t=%.2fs, PLA_prev=%.2f ──\n', k, S.t(i+1), pla_prev);
  fprintf('  %-22s %10s %10s %10s %10s %12s\n', ...
          '策略','首步PLA','预测峰值','预测超调','末值误差','等权MSE');
  fprintf('  %s\n', repmat('-',1,80));

  best_mse = inf; best_name = '';
  for si = 1:numel(STRATS)
    st = STRATS{si};
    pla = min(78, max(48, st.pla));
    % 预测
    obs_z = (obs_h - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
    act_z = (act_h - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
    af = zeros(1,H,3);
    af(1,:,1) = (pla - P.act_mu(1)) / P.act_sd(1);
    af(1,:,2) = (0 - P.act_mu(2)) / P.act_sd(2);
    af(1,:,3) = (0 - P.act_mu(3)) / P.act_sd(3);
    pr = roll_b(obs_z, act_z, af, P);
    n2p = pr(1,:,2) * P.obs_sd(2) + P.obs_mu(2);
    e = n2p - N2_REF;
    mse = mean(e.^2);
    pk = max(n2p);
    fprintf('  %-22s %10.2f %10.1f %10.1f %10.1f %12.1f\n', ...
            st.name, pla(1), pk, pk-N2_REF, e(end), mse);
    if mse < best_mse, best_mse = mse; best_name = st.name; end
  end

  % ══ CEM 解（从 expB 载入或重算）══
  CFG = struct('H',H,'n_cand',32,'n_iter',6,'n_short',8,'w_short',3.0, ...
    'w_n2',1.0,'w_du',1.0,'w_coast',0.0, ...
    'pla_min',48,'pla_max',78,'dpla_max',2.0,'warm_start',false,'mu_prev',[]);
  [pla_cem, info] = mpc_opt_diag(obs_h, act_h, N2_REF, pla_prev, CFG, P, k*17);
  e_cem = info.pred_n2_mu - N2_REF;
  mse_cem = mean(e_cem.^2);
  fprintf('  %-22s %10.2f %10.1f %10.1f %10.1f %12.1f\n', ...
          '★CEM 实际选择', pla_cem(1), max(info.pred_n2_mu), ...
          max(info.pred_n2_mu)-N2_REF, e_cem(end), mse_cem);

  fprintf('  -> 最优策略: %s (MSE=%.1f)\n', best_name, best_mse);
  if mse_cem > best_mse * 1.05
    fprintf('  -> ❌ CEM 解比最优策略差 %.1f%%（探索不足）\n', ...
            (mse_cem/best_mse-1)*100);
  else
    fprintf('  -> ✅ CEM 解接近最优\n');
  end
  fprintf('\n');
end

fprintf('PHYS_OK\n');

function pred = roll_b(obs_z, act_z, af, P)
H = size(af,2);
[hh, ~] = enc_h(obs_z, act_z, P);
o_prev = obs_z(end,:);
pred = zeros(1,H,5);
for k=1:H
  afk = reshape(af(:,k,:),1,3);
  z=[o_prev,afk]*P.w_o2h_weight' + reshape(P.w_o2h_bias,1,[]);
  hh=gs(z,hh,P.w_cell_weight_ih,P.w_cell_weight_hh,P.w_cell_bias_ih,P.w_cell_bias_hh);
  a1=hh*P.w_dec_0_weight'+reshape(P.w_dec_0_bias,1,[]); a1=a1.*(1./(1+exp(-a1)));
  a2=a1*P.w_dec_2_weight'+reshape(P.w_dec_2_bias,1,[]); a2=a2.*(1./(1+exp(-a2)));
  o=a2*P.w_dec_4_weight'+reshape(P.w_dec_4_bias,1,[]);
  pred(1,k,:)=o; o_prev=o;
end
end

function [hh,h3]=enc_h(obs_z,act_z,P)
W=size(obs_z,1); x=[obs_z,act_z];
h1=x*P.w_inp_0_weight'+reshape(P.w_inp_0_bias,1,[]);
mu=mean(h1,2); va=mean((h1-mu).^2,2);
h2=(h1-mu)./sqrt(va+1e-5).*reshape(P.w_inp_1_weight,1,[])+reshape(P.w_inp_1_bias,1,[]);
h3=h2.*(1./(1+exp(-h2))); hh=zeros(1,128);
for t=1:W
  hh=gs(h3(t,:),hh,P.w_gru_weight_ih_l0,P.w_gru_weight_hh_l0,P.w_gru_bias_ih_l0,P.w_gru_bias_hh_l0);
end
end

function h=gs(x,h,wih,whh,bih,bhh)
x=reshape(x,1,[]); h=reshape(h,1,[]); bih=reshape(bih,1,[]); bhh=reshape(bhh,1,[]);
Hg=size(whh,1)/3;
xr=x*wih(1:Hg,:)'+bih(1:Hg); xz=x*wih(Hg+1:2*Hg,:)'+bih(Hg+1:2*Hg); xn=x*wih(2*Hg+1:3*Hg,:)'+bih(2*Hg+1:3*Hg);
hr=h*whh(1:Hg,:)'+bhh(1:Hg); hz=h*whh(Hg+1:2*Hg,:)'+bhh(Hg+1:2*Hg); hn=h*whh(2*Hg+1:3*Hg,:)'+bhh(2*Hg+1:3*Hg);
r=1./(1+exp(-(xr+hr))); z=1./(1+exp(-(xz+hz))); nn=tanh(xn+r.*hn); h=(1-z).*nn+z.*h;
end

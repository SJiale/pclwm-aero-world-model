function ablation_mpc()
% 任务1 消融：用 GRU 自闭环做离线扫描（快速）
%
% 注意：这是"模型自洽"消融，不是真实环境验证
%       真实闭环放到 mpc_run3.m 做
%
% 消融变量：
%   - w_du（动作惩罚）
%   - warm_start（是否用上一步解初始化）
%   - dpla_max（变化率限制）
%   - 有无目标函数前段加权

addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');
outDir = 'C:\pclwm\pclwm_stage3a\iiic';
outfid = fopen(fullfile(outDir,'ablation.txt'),'w','n','UTF-8');

fprintf(outfid,'=== MPC 消融实验（GRU 自闭环）===\n');
fprintf(outfid,'  注意：这是模型自洽消融，用于比较配置相对优劣\n\n');

% 起始状态
obs_steady = [0.6180, 4349.29, 2465.17, 1715.78, 303.61];
n2_steady = 4349.29;
N_CL = 40;
N2_REF_OFF = 50.0;
i_ref = 3;

BASE = struct();
BASE.H = 32; BASE.n_cand = 32; BASE.n_iter = 4;
BASE.n_short = 8; BASE.w_short = 3.0;
BASE.w_n2 = 1.0; BASE.w_du = 0.1; BASE.w_coast = 0.0;
BASE.pla_min = 50; BASE.pla_max = 60; BASE.dpla_max = 1.0;
BASE.warm_start = false; BASE.mu_prev = [];

CASES = {};
% 基线（III-A 配置）
c = BASE; c.name = 'baseline'; CASES{end+1}=c;
% 动作惩罚
c = BASE; c.name = 'w_du=1';     c.w_du = 1.0;    CASES{end+1}=c;
c = BASE; c.name = 'w_du=10';    c.w_du = 10.0;   CASES{end+1}=c;
c = BASE; c.name = 'w_du=100';   c.w_du = 100.0;  CASES{end+1}=c;
% warm-start
c = BASE; c.name = 'warm_start'; c.warm_start = true; CASES{end+1}=c;
% 变化率
c = BASE; c.name = 'dpla=0.3';   c.dpla_max = 0.3; CASES{end+1}=c;
c = BASE; c.name = 'dpla=2.0';   c.dpla_max = 2.0; CASES{end+1}=c;
% 前段加权
c = BASE; c.name = 'no_short_w'; c.w_short = 1.0;  CASES{end+1}=c;
% 更多迭代
c = BASE; c.name = 'iter=6';     c.n_iter = 6;     CASES{end+1}=c;

fprintf(outfid,'  %-14s %10s %10s %10s %10s %10s\n', ...
        '配置','MAE(rpm)','末步err','超调','Σ|ΔPLA|','抖动std');
fprintf(outfid,'  %s\n', repmat('-',1,72));

RESULTS = [];
for ci = 1:numel(CASES)
  cf = CASES{ci};
  obs_hist = repmat(obs_steady, 32, 1);
  act_hist = repmat([55,0,0], 32, 1);
  pla_prev = 55.0;
  mu_prev = [];
  log = zeros(N_CL, 5);   % [step, n2, ref, pla, err]

  for k = 1:N_CL
    n2_cur = obs_hist(end,2);
    n2_ref = n2_steady + N2_REF_OFF*(k>=i_ref);

    cf_k = cf;
    cf_k.mu_prev = mu_prev;
    [pla_seq, info] = mpc_opt(obs_hist, act_hist, n2_ref, pla_prev, cf_k, P, k*17);
    pla_apply = pla_seq(1);
    mu_prev = info.mu;
    pla_prev = pla_apply;

    % 环境推进（用 GRU 单步，非批量）
    obs_z = (obs_hist - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
    act_z = (act_hist - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
    [h0,~] = enc_h(obs_z, act_z, P);
    af = zeros(1,1,3);
    af(1,1,1) = (pla_apply - P.act_mu(1))/P.act_sd(1);
    af(1,1,2) = (0 - P.act_mu(2))/P.act_sd(2);
    af(1,1,3) = (0 - P.act_mu(3))/P.act_sd(3);
    pr = roll_b(h0, obs_z(end,:), af, P);
    obs_new_z = squeeze(pr(1,1,:))';
    obs_new = obs_new_z .* P.obs_sd + P.obs_mu;

    obs_hist = [obs_hist(2:end,:); obs_new];
    act_hist = [act_hist(2:end,:); [pla_apply,0,0]];
    log(k,:) = [k, obs_new(2), n2_ref, pla_apply, obs_new(2)-n2_ref];
  end

  e = log(i_ref:end, 5);
  mae = mean(abs(e));
  final_e = abs(e(end));
  overshoot = max(0, max(e));
  sum_dpla = sum(abs(diff(log(:,4))));
  jitter = std(diff(log(:,4)));

  fprintf(outfid,'  %-14s %10.2f %10.2f %10.2f %10.3f %10.4f\n', ...
          cf.name, mae, final_e, overshoot, sum_dpla, jitter);
  RESULTS(end+1,:) = [mae, final_e, overshoot, sum_dpla, jitter];
end

fprintf(outfid,'\n[解读]\n');
fprintf(outfid,'  MAE / 末步err / 超调 越小越好（跟踪）\n');
fprintf(outfid,'  Σ|ΔPLA| / 抖动std 越小越好（平滑）\n');

% 找 Pareto 前沿
fprintf(outfid,'\n[Pareto 分析] 跟踪 vs 平滑\n');
fprintf(outfid,'  %-14s %10s %10s %6s\n','配置','MAE','Σ|ΔPLA|','Pareto');
for ci=1:numel(CASES)
  mae = RESULTS(ci,1); sd = RESULTS(ci,4);
  dom = false;
  for cj=1:numel(CASES)
    if cj~=ci && RESULTS(cj,1)<=mae && RESULTS(cj,4)<=sd && ...
       (RESULTS(cj,1)<mae || RESULTS(cj,4)<sd)
      dom = true; break;
    end
  end
  fprintf(outfid,'  %-14s %10.2f %10.3f %6s\n', CASES{ci}.name, mae, sd, ...
          ternary(~dom,'✓',''));
end

fclose(outfid);
fprintf('PHYS_OK\n');
end

function s = ternary(c,a,b)
if c, s=a; else, s=b; end
end

function [h_last,h3] = enc_h(obs_z,act_z,P)
W=size(obs_z,1); x=[obs_z,act_z];
h1=x*P.w_inp_0_weight'+reshape(P.w_inp_0_bias,1,[]);
mu=mean(h1,2); va=mean((h1-mu).^2,2);
h2=(h1-mu)./sqrt(va+1e-5).*reshape(P.w_inp_1_weight,1,[])+reshape(P.w_inp_1_bias,1,[]);
h3=h2.*(1./(1+exp(-h2))); hh=zeros(1,128);
for t=1:W
  hh=gs(h3(t,:),hh,P.w_gru_weight_ih_l0,P.w_gru_weight_hh_l0,P.w_gru_bias_ih_l0,P.w_gru_bias_hh_l0);
end
h_last=hh;
end
function h=gs(x,h,wih,whh,bih,bhh)
x=reshape(x,1,[]); h=reshape(h,1,[]); bih=reshape(bih,1,[]); bhh=reshape(bhh,1,[]);
Hg=size(whh,1)/3;
xr=x*wih(1:Hg,:)'+bih(1:Hg); xz=x*wih(Hg+1:2*Hg,:)'+bih(Hg+1:2*Hg); xn=x*wih(2*Hg+1:3*Hg,:)'+bih(2*Hg+1:3*Hg);
hr=h*whh(1:Hg,:)'+bhh(1:Hg); hz=h*whh(Hg+1:2*Hg,:)'+bhh(Hg+1:2*Hg); hn=h*whh(2*Hg+1:3*Hg,:)'+bhh(2*Hg+1:3*Hg);
r=1./(1+exp(-(xr+hr))); z=1./(1+exp(-(xz+hz))); nn=tanh(xn+r.*hn); h=(1-z).*nn+z.*h;
end
function pred=roll_b(h0,o0,af,P)
B=size(af,1); H=size(af,2); h=repmat(h0,B,1); o_prev=repmat(o0,B,1); pred=zeros(B,H,5);
for k=1:H
  afk=reshape(af(:,k,:),B,3);
  z=[o_prev,afk]*P.w_o2h_weight'+repmat(reshape(P.w_o2h_bias,1,[]),B,1);
  h=gsb(z,h,P.w_cell_weight_ih,P.w_cell_weight_hh,P.w_cell_bias_ih,P.w_cell_bias_hh);
  a1=h*P.w_dec_0_weight'+repmat(reshape(P.w_dec_0_bias,1,[]),B,1); a1=a1.*(1./(1+exp(-a1)));
  a2=a1*P.w_dec_2_weight'+repmat(reshape(P.w_dec_2_bias,1,[]),B,1); a2=a2.*(1./(1+exp(-a2)));
  o=a2*P.w_dec_4_weight'+repmat(reshape(P.w_dec_4_bias,1,[]),B,1); pred(:,k,:)=o; o_prev=o;
end
end
function h=gsb(x,h,wih,whh,bih,bhh)
Hg=size(whh,1)/3;
xr=x*wih(1:Hg,:)'+reshape(bih(1:Hg),1,[]); xz=x*wih(Hg+1:2*Hg,:)'+reshape(bih(Hg+1:2*Hg),1,[]); xn=x*wih(2*Hg+1:3*Hg,:)'+reshape(bih(2*Hg+1:3*Hg),1,[]);
hr=h*whh(1:Hg,:)'+reshape(bhh(1:Hg),1,[]); hz=h*whh(Hg+1:2*Hg,:)'+reshape(bhh(Hg+1:2*Hg),1,[]); hn=h*whh(2*Hg+1:3*Hg,:)'+reshape(bhh(2*Hg+1:3*Hg),1,[]);
r=1./(1+exp(-(xr+hr))); z=1./(1+exp(-(xz+hz))); nn=tanh(xn+r.*hn); h=(1-z).*nn+z.*h;
end



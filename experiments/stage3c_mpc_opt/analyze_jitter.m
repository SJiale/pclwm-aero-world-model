function analyze_jitter()
% 任务1（诊断部分）：分析 PLA 抖动与"先降后升"的根因
%
% 疑点：
%   1. 第3步 PLA 先降到 54.16（应该升才对）
%   2. PLA 序列抖动（54.16→57.41→55.03）
%
% 检查：
%   A. CEM 的解是否稳定（同输入多次运行）
%   B. 目标函数在最优解附近是否平坦
%   C. 模型预测偏差（模型认为降 PLA 会让 N2 升？）

addpath('C:\pclwm\pclwm_stage3a');
P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');
outDir = 'C:\pclwm\pclwm_stage3a\iiic';
outfid = fopen(fullfile(outDir,'jitter_diag.txt'),'w','n','UTF-8');

fprintf(outfid,'=== PLA 抖动与先降后升诊断 ===\n\n');

obs_steady = [0.6180, 4349.29, 2465.17, 1715.78, 303.61];
obs_z = (repmat(obs_steady,32,1) - reshape(P.obs_mu,1,[])) ./ reshape(P.obs_sd,1,[]);
act_z = (repmat([55,0,0],32,1) - reshape(P.act_mu,1,[])) ./ reshape(P.act_sd,1,[]);
H = 32;
n2_now = 4349.29;
n2_ref = n2_now + 50;

% ══════════════════════════════════════════════════
% [A] 模型预测：PLA 常值 → 末步 N2
% ══════════════════════════════════════════════════
fprintf(outfid,'[A] 模型预测：PLA 常值激励\n');
fprintf(outfid,'  %10s %14s %14s\n','PLA','步1 N2','步32 N2');
[h0,~] = encode_history(obs_z, act_z, P);
for pla_v = [52 54 55 56 58]
  af = zeros(1,H,3);
  af(1,:,1) = (pla_v - P.act_mu(1))/P.act_sd(1);
  af(1,:,2) = (0 - P.act_mu(2))/P.act_sd(2);
  af(1,:,3) = (0 - P.act_mu(3))/P.act_sd(3);
  pr = gru_rollout_batch(h0, obs_z(end,:), af, P);
  n2p = squeeze(pr(1,:,2))*P.obs_sd(2) + P.obs_mu(2);
  fprintf(outfid,'  %10.1f %14.3f %14.3f\n', pla_v, n2p(1), n2p(end));
end

% ══════════════════════════════════════════════════
% [B] 目标函数形状：固定 PLA 全程 → J
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[B] 目标函数：PLA 全程常值\n');
fprintf(outfid,'  %10s %14s\n','PLA','J (全程)');
Jbest = inf; PLAbest = NaN;
for pla_v = 50:0.1:60
  af = zeros(1,H,3);
  af(1,:,1) = (pla_v - P.act_mu(1))/P.act_sd(1);
  af(1,:,2) = (0 - P.act_mu(2))/P.act_sd(2);
  af(1,:,3) = (0 - P.act_mu(3))/P.act_sd(3);
  pr = gru_rollout_batch(h0, obs_z(end,:), af, P);
  n2p = squeeze(pr(1,:,2))*P.obs_sd(2) + P.obs_mu(2);
  J = sum((n2p - n2_ref).^2);
  if J < Jbest, Jbest = J; PLAbest = pla_v; end
end
fprintf(outfid,'  最优常值 PLA = %.2f (J=%.1f)\n', PLAbest, Jbest);
fprintf(outfid,'  -> 若 >55，说明需要升 PLA\n');

% ══════════════════════════════════════════════════
% [C] CEM 多次运行的解稳定性
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[C] CEM 解稳定性（同输入，不同种子，20 次）\n');
sols = zeros(20, H);
for r = 1:20
  [pla_seq, ~] = cem_diag(obs_z, act_z, n2_ref, 55.0, 32, 4, H, P, r*7);
  sols(r,:) = pla_seq;
end
fprintf(outfid,'  PLA[1] (第一步) 统计:\n');
fprintf(outfid,'    中位=%.3f  均值=%.3f  std=%.4f  [min,max]=[%.3f, %.3f]\n', ...
        median(sols(:,1)), mean(sols(:,1)), std(sols(:,1)), ...
        min(sols(:,1)), max(sols(:,1)));
fprintf(outfid,'  PLA[1] < 55 的比例: %.0f%%\n', sum(sols(:,1)<55)/20*100);
fprintf(outfid,'  PLA 序列末步统计: 中位=%.3f std=%.4f\n', ...
        median(sols(:,end)), std(sols(:,end)));

% 序列抖动幅度
jit = mean(std(sols, 0, 1));
fprintf(outfid,'  跨运行的平均标准差（逐时间步）: %.4f deg\n', jit);

% ══════════════════════════════════════════════════
% [D] 加入动作惩罚的效果
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[D] 动作惩罚对抖动的影响\n');
fprintf(outfid,'  %10s %14s %14s %14s\n','w_du','PLA[1]中位','抖动(std)','J_track');
for w_du = [0 0.1 1 10 100]
  sols2 = zeros(20,H); Jt = zeros(20,1);
  for r = 1:20
    [ps, jt] = cem_diag2(obs_z, act_z, n2_ref, 55.0, 32, 4, H, P, r*7, w_du);
    sols2(r,:) = ps; Jt(r) = jt;
  end
  fprintf(outfid,'  %10.1f %14.3f %14.4f %14.1f\n', w_du, ...
          median(sols2(:,1)), mean(std(sols2,0,1)), median(Jt));
end

fclose(outfid);
fprintf('PHYS_OK\n');
end

% ══ 辅助 ══
function [h_last, h3] = encode_history(obs_z, act_z, P)
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
function pred = gru_rollout_batch(h0,o_prev0,af_all,P)
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
function [pla, Jt] = cem_diag(obs_z,act_z,n2_ref,pla0,nc,ni,H,P,seed)
[pla, Jt] = cem_diag2(obs_z,act_z,n2_ref,pla0,nc,ni,H,P,seed,0.0);
end
function [pla, Jt] = cem_diag2(obs_z,act_z,n2_ref,pla0,nc,ni,H,P,seed,w_du)
rng(seed);
[h0,~]=encode_history(obs_z,act_z,P); o0=obs_z(end,:);
mu=pla0*ones(1,H); sig=0.5*ones(1,H); n_el=max(2,floor(nc/4));
Jt=inf;
for it=1:ni
  cand=mu+sig.*randn(nc,H); cand=min(60,max(50,cand));
  af=zeros(nc,H,3);
  af(:,:,1)=(cand-P.act_mu(1))/P.act_sd(1);
  af(:,:,2)=(0-P.act_mu(2))/P.act_sd(2);
  af(:,:,3)=(0-P.act_mu(3))/P.act_sd(3);
  pr=gru_rollout_batch(h0,o0,af,P);
  n2p=pr(:,:,2)*P.obs_sd(2)+P.obs_mu(2);
  jt=sum((n2p-n2_ref).^2,2);
  du=zeros(nc,1);
  for c=1:nc
    du(c)=sum(diff([pla0,cand(c,:)]).^2);
  end
  J=jt+w_du*du;
  [Js,idx]=sort(J);
  if w_du==0 && Js(1)<Jt, Jt=Js(1); end
  if w_du>0
    [~,i2]=sort(jt); Jt=median(jt(i2(1:n_el)));
  end
  el=cand(idx(1:n_el),:); mu=mean(el,1); sig=std(el,0,1)+1e-3;
end
pla=min(60,max(50,mu));
end

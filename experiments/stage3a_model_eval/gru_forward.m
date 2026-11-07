function pred = gru_forward(obs_z, act_z, af_z, P)
% GRU 前向（与 PyTorch 一致，max|diff|~8e-7）
%
% 输入：
%   obs_z : (W,5) 归一化历史观测
%   act_z : (W,3) 归一化历史动作
%   af_z  : (H,3) 归一化未来动作
% 输出：
%   pred  : (H,5) 归一化预测

W = size(obs_z, 1);
Hh = size(af_z, 1);

% ── 历史编码 ──
x = [obs_z, act_z];                                    % (W,8)
h1 = x * P.w_inp_0_weight' + reshape(P.w_inp_0_bias,1,[]);
mu = mean(h1, 2);
va = mean((h1 - mu).^2, 2);
h2 = (h1 - mu) ./ sqrt(va + 1e-5) .* reshape(P.w_inp_1_weight,1,[]) ...
     + reshape(P.w_inp_1_bias,1,[]);
h3 = h2 .* (1 ./ (1 + exp(-h2)));                      % (W,128)

% ── GRU 序列 ──
hh = zeros(1, 128);
for t = 1:W
  hh = gru_step(h3(t,:), hh, P.w_gru_weight_ih_l0, P.w_gru_weight_hh_l0, ...
                P.w_gru_bias_ih_l0, P.w_gru_bias_hh_l0);
end

% ── 递推未来 ──
o_prev = obs_z(end, :);
pred = zeros(Hh, 5);
for k = 1:Hh
  z_in = [o_prev, af_z(k,:)];
  z = z_in * P.w_o2h_weight' + reshape(P.w_o2h_bias,1,[]);
  hh = gru_step(z, hh, P.w_cell_weight_ih, P.w_cell_weight_hh, ...
                P.w_cell_bias_ih, P.w_cell_bias_hh);
  a1 = hh * P.w_dec_0_weight' + reshape(P.w_dec_0_bias,1,[]);
  a1 = a1 .* (1 ./ (1 + exp(-a1)));
  a2 = a1 * P.w_dec_2_weight' + reshape(P.w_dec_2_bias,1,[]);
  a2 = a2 .* (1 ./ (1 + exp(-a2)));
  o_hat = a2 * P.w_dec_4_weight' + reshape(P.w_dec_4_bias,1,[]);
  pred(k,:) = o_hat;
  o_prev = o_hat;
end
end

function h = gru_step(x, h, wih, whh, bih, bhh)
x = reshape(x,1,[]);  h = reshape(h,1,[]);
bih = reshape(bih,1,[]);  bhh = reshape(bhh,1,[]);
Hg = size(whh, 1) / 3;
xr = x * wih(1:Hg, :)'       + bih(1:Hg);
xz = x * wih(Hg+1:2*Hg, :)'  + bih(Hg+1:2*Hg);
xn = x * wih(2*Hg+1:3*Hg, :)'+ bih(2*Hg+1:3*Hg);
hr = h * whh(1:Hg, :)'       + bhh(1:Hg);
hz = h * whh(Hg+1:2*Hg, :)'  + bhh(Hg+1:2*Hg);
hn = h * whh(2*Hg+1:3*Hg, :)'+ bhh(2*Hg+1:3*Hg);

r = 1 ./ (1 + exp(-(xr + hr)));
z = 1 ./ (1 + exp(-(xz + hz)));
n = tanh(xn + r .* hn);
h = (1 - z) .* n + z .* h;
end

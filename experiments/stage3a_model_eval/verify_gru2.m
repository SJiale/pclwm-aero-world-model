function verify_gru2()
% 验证 MATLAB GRU 前向（使用 gru_pack2.mat）
%
% 所有向量为行向量 (1×N)
% Linear: y = x*W' + b,  x:(1,in), W:(out,in), W':(in,out) -> y:(1,out)

P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');
fprintf('=== 验证 MATLAB GRU（v2）===\n');

obs = P.tc_obs;       % (32,5)
act = P.tc_act;       % (32,3)
af = P.tc_af;         % (32,3)
expected = P.tc_expected;  % (32,5)

% ── 1. 历史编码 ──
x = [obs, act];                                    % (32,8)
h1 = x * P.w_inp_0_weight' + reshape(P.w_inp_0_bias,1,[]);
% LayerNorm
mu = mean(h1, 2);                                  % (32,1)
va = mean((h1 - mu).^2, 2);                        % (32,1)
h2 = (h1 - mu) ./ sqrt(va + 1e-5) .* reshape(P.w_inp_1_weight,1,[]) + reshape(P.w_inp_1_bias,1,[]);
h3 = h2 .* (1 ./ (1 + exp(-h2)));                  % SiLU -> (32,128)

% ── 2. GRU 序列扫描 ──
hh = zeros(1, 128);
for t = 1:size(h3, 1)
  hh = gru_step(h3(t, :), hh(:)', P.w_gru_weight_ih_l0, P.w_gru_weight_hh_l0, ...
                P.w_gru_bias_ih_l0, P.w_gru_bias_hh_l0);
end
fprintf('  GRU 最终隐状态: (1,%d)\n', numel(hh));

% ── 3. 递推未来 ──
o_prev = obs(end, :);       % (1,5)
Hh = size(af, 1);
outs = zeros(Hh, 5);
for k = 1:Hh
  z_in = [o_prev, af(k, :)];                       % (1,8)
  z = z_in * P.w_o2h_weight' + reshape(P.w_o2h_bias,1,[]);
  hh = gru_step(z, hh(:)', P.w_cell_weight_ih, P.w_cell_weight_hh, ...
                P.w_cell_bias_ih, P.w_cell_bias_hh);
  % dec: 128 -> 256 -> 256 -> 5
  a1 = hh * P.w_dec_0_weight' + reshape(P.w_dec_0_bias,1,[]);
  a1 = a1 .* (1 ./ (1 + exp(-a1)));
  a2 = a1 * P.w_dec_2_weight' + reshape(P.w_dec_2_bias,1,[]);
  a2 = a2 .* (1 ./ (1 + exp(-a2)));
  o_hat = a2 * P.w_dec_4_weight' + reshape(P.w_dec_4_bias,1,[]);
  outs(k, :) = o_hat;
  o_prev = o_hat;
end

% ── 4. 对比 ──
d = abs(outs - expected);
fprintf('\n  输出对比（前3步）:\n');
for k = 1:3
  fprintf('    step %d\n', k);
  fprintf('      MATLAB : %s\n', sprintf('%8.5f ', outs(k,:)));
  fprintf('      PyTorch: %s\n', sprintf('%8.5f ', expected(k,:)));
end
fprintf('\n  max|diff| = %.3e\n', max(d(:)));
fprintf('  mean|diff| = %.3e\n', mean(d(:)));
if max(d(:)) < 1e-4
  fprintf('  -> ✅ 实现正确\n');
else
  fprintf('  -> ❌ 有偏差\n');
end

% 保存结果供后续使用
save('C:\pclwm\pclwm_stage3a\gru_verify_result.mat', 'outs', 'expected');
end

% ══════════════════════════════════════════════════
function h = gru_step(x, h, wih, whh, bih, bhh)
% x:(1,in), h:(1,H), wih:(3H,in), whh:(3H,H)
% ★ bias 在 mat 里是 (1,N) 行向量，统一 reshape 成行
x = reshape(x, 1, []);  h = reshape(h, 1, []);
bih = reshape(bih, 1, []);  bhh = reshape(bhh, 1, []);
H = size(whh, 1) / 3;
xr = x * wih(1:H, :)'      + bih(1:H);
xz = x * wih(H+1:2*H, :)'  + bih(H+1:2*H);
xn = x * wih(2*H+1:3*H, :)'+ bih(2*H+1:3*H);
hr = h * whh(1:H, :)'      + bhh(1:H);
hz = h * whh(H+1:2*H, :)'  + bhh(H+1:2*H);
hn = h * whh(2*H+1:3*H, :)'+ bhh(2*H+1:3*H);

r = 1 ./ (1 + exp(-(xr + hr)));
z = 1 ./ (1 + exp(-(xz + hz)));
n = tanh(xn + r .* hn);
h = (1 - z) .* n + z .* h;
end



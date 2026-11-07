% 检查 causal_diag.mat 里 mu 的真实结构
CD = load('C:\pclwm\pclwm_stage3a\causal\causal_diag.mat', 'DIAG');
DIAG = CD.DIAG;

fprintf('=== DIAG 结构 ===\n');
fprintf('  n_steps = %d\n', DIAG.n_steps);
fprintf('  steps 长度 = %d\n\n', numel(DIAG.steps));

s = DIAG.steps{10};   % k=10
fprintf('=== k=10 的字段 ===\n');
fn = fieldnames(s);
for i=1:numel(fn)
  v = s.(fn{i});
  if ischar(v) || isstring(v)
    fprintf('  %-20s char\n', fn{i});
  elseif isnumeric(v)
    fprintf('  %-20s %s %s\n', fn{i}, class(v), mat2str(size(v)));
  elseif isstruct(v)
    fprintf('  %-20s struct [%s]\n', fn{i}, mat2str(size(v)));
  elseif iscell(v)
    fprintf('  %-20s cell [%s]\n', fn{i}, mat2str(size(v)));
  else
    fprintf('  %-20s %s\n', fn{i}, class(v));
  end
end

fprintf('\n=== iter_log 结构 ===\n');
il = s.iter_log;
fprintf('  类型 %s, 长度 %d\n', class(il), numel(il));
if numel(il) > 0
  e1 = il(1);
  fprintf('  第1轮字段: %s\n', strjoin(fieldnames(e1)', ', '));
  fprintf('    it     = %d\n', e1.it);
  fprintf('    J_best = %.2f\n', e1.J_best);
  fprintf('    mu     = %s\n', mat2str(e1.mu(1:min(8,end)), 4));
  fprintf('    pred_n2= %s\n', mat2str(e1.pred_n2(1:min(8,end)), 6));
end

fprintf('\n=== 最后一轮 mu ===\n');
el = il(end);
fprintf('  前8: %s\n', mat2str(el.mu(1:8), 4));
fprintf('  后8: %s\n', mat2str(el.mu(end-7:end), 4));

fprintf('\n=== info.mu（最终）===\n');
if isfield(s, 'iter_log') && isfield(s,'mu')
  fprintf('  %s\n', mat2str(s.mu(1:8), 4));
end

fprintf('\n=== pla_seq_full ===\n');
fprintf('  前8: %s\n', mat2str(s.pla_seq_full(1:8), 4));
fprintf('  后8: %s\n', mat2str(s.pla_seq_full(end-7:end), 4));

% 检查 DIAG.steps{k}.mu 是否存在
fprintf('\n=== 各步 mu 多样性检查 ===\n');
for k=[10 14 17]
  ss = DIAG.steps{k};
  il2 = ss.iter_log;
  m_last = il2(end).mu;
  fprintf('  k=%d: iter_log(end).mu 前5 = %s, std = %.4f\n', ...
    k, mat2str(m_last(1:5),4), std(m_last));
end

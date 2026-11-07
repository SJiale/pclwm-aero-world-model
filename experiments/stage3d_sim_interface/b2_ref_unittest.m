% B2 参考信号单元测试
%
% 设计：
%   N2_0 = 4349.29 rpm
%   0.0 - 6.3s : N2_ref = N2_0 + 750 = 5099.29
%   6.3 - 12.6s: N2_ref = N2_0 + 250 = 4599.29
%
% 120 步 x 0.105s = 12.6s
%
% 断言：
%   A1. 参考值只包含两个平台值
%   A2. 阶跃发生在第 61 步（t=6.30s）
%   A3. 跳变量为 +750（0→6.3）和 -500（6.3→12.6）
%   A4. 无多余跳变

outDir = 'C:\pclwm\pclwm_stage3a\iiid';
if ~isfolder(outDir), mkdir(outDir); end

fid = fopen(fullfile(outDir,'ref_unittest.txt'),'w','n','UTF-8');

fprintf(fid,'=== B2 参考信号单元测试 ===\n\n');

% ══ 参数 ══
N2_0 = 4349.29;
CTRL_DT = 0.105;
N_CL = 120;

T_SEG1_END = 6.3;
T_SEG2_END = 12.6;
DELTA1 = +750;    % 第一段
DELTA2 = +250;    % 第二段（相对初始，等价于从峰值 -500）

REF1 = N2_0 + DELTA1;   % 5099.29
REF2 = N2_0 + DELTA2;   % 4599.29

fprintf(fid,'参数:\n');
fprintf(fid,'  N2_0     = %.2f rpm\n', N2_0);
fprintf(fid,'  REF1     = N2_0 + %d = %.2f rpm\n', DELTA1, REF1);
fprintf(fid,'  REF2     = N2_0 + %d = %.2f rpm  (相对 REF1 为 %+d)\n', ...
        DELTA2, REF2, DELTA2-DELTA1);
fprintf(fid,'  CTRL_DT  = %.3f s\n', CTRL_DT);
fprintf(fid,'  N_CL     = %d 步 (%.2f s)\n\n', N_CL, N_CL*CTRL_DT);

% ══ 生成参考数组 ══
% 每个控制步 k（1..N_CL）对应的时刻是 (k-1)*CTRL_DT
t_step = ((0:N_CL-1)') * CTRL_DT;
n2_ref = zeros(N_CL, 1);
for k = 1:N_CL
  tk = t_step(k);
  if tk < T_SEG1_END
    n2_ref(k) = REF1;
  else
    n2_ref(k) = REF2;
  end
end

fprintf(fid,'参考数组前 3 步和后 3 步:\n');
for k=1:3
  fprintf(fid,'  k=%3d  t=%6.3f  ref=%.2f\n', k, t_step(k), n2_ref(k));
end
fprintf(fid,'  ...\n');
for k=N_CL-2:N_CL
  fprintf(fid,'  k=%3d  t=%6.3f  ref=%.2f\n', k, t_step(k), n2_ref(k));
end

% ══ 断言 ══
fprintf(fid,'\n=== 断言 ===\n');
n_pass = 0; n_fail = 0;

% A1. 只包含两个平台值
uniq = unique(n2_ref);
tol = 1e-6;
a1 = (numel(uniq) == 2) && ...
     (abs(uniq(1) - REF2) < tol) && (abs(uniq(2) - REF1) < tol);
fprintf(fid,'  [A1] 只含两个平台值: ');
if a1
  fprintf(fid,'✅ %s\n', mat2str(uniq', 8));
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ %s\n', mat2str(uniq', 8));
  n_fail = n_fail + 1;
end

% A2. 阶跃发生在第 61 步
d = diff(n2_ref);
jump_idx = find(abs(d) > tol);   % 跳变位置（在第 idx 和 idx+1 步之间）
a2 = (numel(jump_idx) == 1) && (jump_idx(1) == 60);
if a2
  fprintf(fid,'  [A2] 阶跃位置: ✅ 第 %d->%d 步 (t=%.3f s)\n', ...
          jump_idx(1), jump_idx(1)+1, t_step(jump_idx(1)+1));
  n_pass = n_pass + 1;
else
  fprintf(fid,'  [A2] 阶跃位置: ❌ 找到 %d 处跳变 %s\n', ...
          numel(jump_idx), mat2str(jump_idx'));
  n_fail = n_fail + 1;
end

% A3. 跳变量
a3 = (numel(jump_idx) == 1) && (abs(d(jump_idx(1)) - (REF2-REF1)) < tol);
if a3
  fprintf(fid,'  [A3] 跳变量: ✅ %+.2f rpm (= REF2-REF1 = %+.2f)\n', ...
          d(jump_idx(1)), REF2-REF1);
  n_pass = n_pass + 1;
else
  fprintf(fid,'  [A3] 跳变量: ❌ %s\n', mat2str(d(jump_idx)', 8));
  n_fail = n_fail + 1;
end

% A4. 第一段必须是 +750（相对 N2_0）
a4 = abs(n2_ref(1) - REF1) < tol;
fprintf(fid,'  [A4] 第 1 步为 REF1 (+750): ');
if a4
  fprintf(fid,'✅ %.2f\n', n2_ref(1));
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ %.2f (应 %.2f)\n', n2_ref(1), REF1);
  n_fail = n_fail + 1;
end

% A5. 第二段必须是 +250（相对 N2_0）
a5 = abs(n2_ref(end) - REF2) < tol;
fprintf(fid,'  [A5] 末步为 REF2 (+250): ');
if a5
  fprintf(fid,'✅ %.2f\n', n2_ref(end));
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ %.2f (应 %.2f)\n', n2_ref(end), REF2);
  n_fail = n_fail + 1;
end

% A6. 第二段相对第一段下降恰好 500
a6 = abs((REF1 - REF2) - 500) < tol;
fprintf(fid,'  [A6] 第二段相对第一段下降 500: ');
if a6
  fprintf(fid,'✅ %.2f\n', REF1 - REF2);
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ %.2f\n', REF1 - REF2);
  n_fail = n_fail + 1;
end

% A7. 不允许出现 3849 附近的值
a7 = all(abs(n2_ref - 3849.29) > 100);
fprintf(fid,'  [A7] 不含错误参考 3849: ');
if a7
  fprintf(fid,'✅\n');
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ 发现 3849 附近的值\n');
  n_fail = n_fail + 1;
end

% A8. 各平台步数
n_s1 = sum(abs(n2_ref - REF1) < tol);
n_s2 = sum(abs(n2_ref - REF2) < tol);
a8 = (n_s1 + n_s2 == N_CL) && (n_s1 == 60) && (n_s2 == 60);
fprintf(fid,'  [A8] 平台步数: ');
if a8
  fprintf(fid,'✅ 段1=%d 步, 段2=%d 步\n', n_s1, n_s2);
  n_pass = n_pass + 1;
else
  fprintf(fid,'❌ 段1=%d 步, 段2=%d 步 (应各 60)\n', n_s1, n_s2);
  n_fail = n_fail + 1;
end

% ══ 汇总 ══
fprintf(fid,'\n=== 结果 ===\n');
fprintf(fid,'  通过 %d, 失败 %d\n', n_pass, n_fail);
if n_fail == 0
  fprintf(fid,'  ✅ 全部通过，允许启动仿真\n');
  fprintf('  ✅ 单元测试全部通过 (%d/%d)\n', n_pass, n_pass);
else
  fprintf(fid,'  ❌ 有失败，禁止启动仿真\n');
  fprintf('  ❌ 单元测试失败 (%d 失败)\n', n_fail);
end

% ══ 保存 ══
save(fullfile(outDir,'b2_ref.mat'), 'n2_ref', 't_step', 'N2_0', ...
     'REF1', 'REF2', 'N_CL', 'CTRL_DT', 'DELTA1', 'DELTA2');
fprintf(fid,'\n  参考数组已保存: b2_ref.mat\n');
fprintf(fid,'    n2_ref: %dx1\n', numel(n2_ref));
fprintf(fid,'    t_step: %dx1\n', numel(t_step));

fclose(fid);
fprintf('PHYS_OK\n');

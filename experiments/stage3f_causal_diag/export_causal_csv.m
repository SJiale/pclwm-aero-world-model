% 把 causal_diag.mat 导出为 CSV（便于 Python 分析）
outDir = 'C:\pclwm\pclwm_stage3a\causal';
load(fullfile(outDir,'causal_diag.mat'), 'DIAG');
N = DIAG.n_steps;
H = DIAG.cfg.H;

% ══ 主表 ══
fid = fopen(fullfile(outDir,'steps.csv'),'w');
fprintf(fid, 'k,t,n2_before,err_before,pla_prev,pla_apply,');
fprintf(fid, 'pred_peak,pred_overshoot,pred_peak_k,pred_peak_t,');
fprintf(fid, 'j_track,j_du,j_coast,J_total,n2_after,wf_after,T45_after,');
fprintf(fid, 'obs_wf,obs_n2,obs_t4,obs_t45,obs_p36\n');
for k=1:N
  s = DIAG.steps{k};
  fprintf(fid, '%d,%.6f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%d,%.6f,%.4f,%.6f,%.6f,%.4f,%.4f,%.6f,%.4f,%.6f,%.4f,%.4f,%.4f,%.4f\n', ...
    s.k, s.t, s.n2_before, s.err_before, s.pla_prev, s.pla_apply, ...
    s.pred_peak_n2, s.pred_overshoot, s.pred_peak_k, s.pred_peak_t, ...
    s.j_track, s.j_du, s.j_coast, s.J_total, s.n2_after, s.wf_after, s.T45_after, ...
    s.obs_before(1), s.obs_before(2), s.obs_before(3), s.obs_before(4), s.obs_before(5));
end
fclose(fid);
fprintf('  已导出 steps.csv (%d 行)\n', N);

% ══ 预测轨迹矩阵 (N x H) ══
P = zeros(N, H);
PS = zeros(N, H);
for k=1:N
  P(k,:) = DIAG.steps{k}.pred_n2_mu;
  PS(k,:) = DIAG.steps{k}.pla_seq_full;
end
writematrix(P, fullfile(outDir,'pred_n2_matrix.csv'));
writematrix(PS, fullfile(outDir,'pla_seq_matrix.csv'));
fprintf('  已导出 pred_n2_matrix.csv (%dx%d)\n', N, H);
fprintf('  已导出 pla_seq_matrix.csv (%dx%d)\n', N, H);

% ══ CEM 迭代日志 ══
fid = fopen(fullfile(outDir,'cem_iters.csv'),'w');
fprintf(fid, 'k,it,J_best,j_track,j_du,pla1,pred_peak\n');
for k=1:N
  il = DIAG.steps{k}.iter_log;
  for j=1:numel(il)
    pn = il(j).pred_n2;
    fprintf(fid, '%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f\n', ...
      DIAG.steps{k}.k, il(j).it, il(j).J_best, il(j).j_track, il(j).j_du, ...
      il(j).mu(1), max(pn));
  end
end
fclose(fid);
fprintf('  已导出 cem_iters.csv\n');

% ══ 配置 ══
fid = fopen(fullfile(outDir,'cfg.txt'),'w');
fprintf(fid, 'n_short=%d\nw_short=%.2f\nH=%d\n', ...
  DIAG.cfg.n_short, DIAG.cfg.w_short, DIAG.cfg.H);
fprintf(fid, 'w_n2=%.2f\nw_du=%.2f\nw_coast=%.4f\n', ...
  DIAG.cfg.w_n2, DIAG.cfg.w_du, DIAG.cfg.w_coast);
fprintf(fid, 'n_cand=%d\nn_iter=%d\n', DIAG.cfg.n_cand, DIAG.cfg.n_iter);
fprintf(fid, 'pla_min=%.1f\npla_max=%.1f\ndpla_max=%.1f\n', ...
  DIAG.cfg.pla_min, DIAG.cfg.pla_max, DIAG.cfg.dpla_max);
fprintf(fid, 'n2_ref=%.2f\nn2_0=%.2f\n', DIAG.n2_ref, DIAG.n2_0);
fclose(fid);
fprintf('  已导出 cfg.txt\n');
fprintf('PHYS_OK\n');

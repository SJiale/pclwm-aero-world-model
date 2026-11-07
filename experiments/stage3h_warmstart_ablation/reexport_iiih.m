% 重新导出干净的 CSV（标签不含逗号，去掉内嵌列表列）
outDir = 'C:\pclwm\pclwm_stage3a\iiih';
CD2 = load(fullfile(outDir,'ablation3.mat'), 'RESULTS');
RESULTS = CD2.RESULTS;

TAGS = {'A', 'B', 'C'};   % 简单标签

% ══ 摘要表 ══
fid = fopen(fullfile(outDir,'ablation3_clean.csv'),'w');
fprintf(fid,'k,t,cfg,pla1,peak,overshoot,peak_k,mse_eq,mae_eq,sum_dpla,err_term,viol_bound,viol_rate\n');
for ki=1:numel(RESULTS)
  for ci=1:3
    e = RESULTS(ki).ent{ci};
    fprintf(fid,'%d,%.4f,%s,%.4f,%.4f,%.4f,%d,%.2f,%.2f,%.4f,%.4f,%d,%d\n', ...
      RESULTS(ki).k, RESULTS(ki).t, TAGS{ci}, e.pla(1), e.peak, e.overshoot, ...
      e.peak_k, e.mse_eq, e.mae_eq, e.sum_dpla, e.err_term, ...
      e.viol_bound, e.viol_rate);
  end
end
fclose(fid);
fprintf('  -> ablation3_clean.csv\n');

% ══ 序列表 ══
fid = fopen(fullfile(outDir,'pla_seq3_clean.csv'),'w');
fprintf(fid,'k,cfg,step,pla,pred_n2\n');
for ki=1:numel(RESULTS)
  for ci=1:3
    e = RESULTS(ki).ent{ci};
    for j=1:numel(e.pla)
      fprintf(fid,'%d,%s,%d,%.4f,%.4f\n', RESULTS(ki).k, TAGS{ci}, j, ...
        e.pla(j), e.pred_n2(j));
    end
  end
end
fclose(fid);
fprintf('  -> pla_seq3_clean.csv\n');

fprintf('PHYS_OK\n');

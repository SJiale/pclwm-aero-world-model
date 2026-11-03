% 导出 Wfact（实际燃油）和 Wfdmd（指令）作为动作
function agtf_export_action()
  dataDir = 'C:\pclwm\agtf30\dataset';
  files = dir(fullfile(dataDir,'r*.mat'));
  files = files(~contains({files.name},'_stats'));

  fprintf('%-34s %10s %10s %8s\n','run','Wf范围','Wfdmd范围','n');
  fprintf('%s\n', repmat('-',1,70));

  for i=1:numel(files)
    [~, base] = fileparts(files(i).name);
    S = load(fullfile(dataDir, files(i).name));
    if ~isfield(S,'out_Dyn'), continue; end
    dyn = S.out_Dyn;

    try
      wfact = dyn.act.Wfact;
      wfdmd = dyn.cntrl.Dmd.Wfdmd;
    catch
      fprintf('%-34s 缺 Wfact/Wfdmd\n', base);
      continue;
    end

    t = wfact.Time(:);
    a1 = wfact.Data(:);
    a2 = wfdmd.Data(:);
    if numel(a2) ~= numel(t)
      a2 = interp1(wfdmd.Time(:), wfdmd.Data(:), t, 'nearest', 'extrap');
    end

    outDir = fullfile(dataDir, base);
    if ~isfolder(outDir), mkdir(outDir); end
    M = [t a1 a2];
    fid2 = fopen(fullfile(outDir,'action.csv'),'w');
    fprintf(fid2,'time_s,Wfact_lb_s,Wfdmd_lb_s\n');
    for r=1:size(M,1)
      fprintf(fid2,'%.10g,%.10g,%.10g\n', M(r,1), M(r,2), M(r,3));
    end
    fclose(fid2);

    fprintf('%-34s %10s %10s %8d\n', base, ...
            sprintf('[%.3f,%.3f]', min(a1), max(a1)), ...
            sprintf('[%.3f,%.3f]', min(a2), max(a2)), numel(t));
  end
  fprintf('\n完成\n');
end

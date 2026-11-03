% 导出 dataset2 的全部数据（obs + action）
function agtf_export2()
  dataDir = 'C:\pclwm\agtf30\dataset2';
  files = dir(fullfile(dataDir,'*.mat'));
  files = files(~contains({files.name},'_stats'));

  blocks = {'out_S0','out_S2','out_S25','out_S36','out_S4','out_S45','out_S5'};
  fields = {'W','Tt','Pt','FAR','ht','Ps','Ts'};

  fprintf('%-40s %6s %8s\n','run','列数','n');
  fprintf('%s\n', repmat('-',1,60));
  nok = 0;
  for i=1:numel(files)
    [~, base] = fileparts(files(i).name);
    S = load(fullfile(dataDir, files(i).name));
    outDir = fullfile(dataDir, base);
    if ~isfolder(outDir), mkdir(outDir); end

    t_ref = []; names = {}; cols = {};
    for b=1:numel(blocks)
      if ~isfield(S, blocks{b}), continue; end
      blk = S.(blocks{b});
      if ~isstruct(blk), continue; end
      for f=1:numel(fields)
        if ~isfield(blk, fields{f}), continue; end
        x = blk.(fields{f});
        if ~isa(x,'timeseries'), continue; end
        t = x.Time(:); d = x.Data;
        if isvector(d), d = d(:); end
        if size(d,2)>1, d = d(:,1); end
        if isempty(t_ref), t_ref = t; end
        if numel(t) ~= numel(t_ref), continue; end
        names{end+1} = sprintf('%s_%s', strrep(blocks{b},'out_',''), fields{f}); %#ok<AGROW>
        cols{end+1} = d; %#ok<AGROW>
      end
    end
    if isempty(cols), continue; end

    % 观测宽表
    M = [t_ref, cell2mat(cols)];
    fid2 = fopen(fullfile(outDir,'data.csv'),'w');
    fprintf(fid2,'%s\n', strjoin(['time_s', names], ','));
    fmt = [repmat('%.10g,',1,size(M,2)-1) '%.10g\n'];
    for r=1:size(M,1), fprintf(fid2,fmt,M(r,:)); end
    fclose(fid2);

    % 动作
    try
      dyn = S.out_Dyn;
      wf = dyn.act.Wfact;
      wd = dyn.cntrl.Dmd.Wfdmd;
      ta = wf.Time(:); a1 = wf.Data(:);
      a2 = interp1(wd.Time(:), wd.Data(:), ta, 'nearest', 'extrap');
      Ma = [ta a1 a2];
      fid3 = fopen(fullfile(outDir,'action.csv'),'w');
      fprintf(fid3,'time_s,Wfact_lb_s,Wfdmd_lb_s\n');
      for r=1:size(Ma,1)
        fprintf(fid3,'%.10g,%.10g,%.10g\n', Ma(r,1), Ma(r,2), Ma(r,3));
      end
      fclose(fid3);
    catch
      fprintf('  %s: action 缺\n', base);
    end

    nok = nok + 1;
    fprintf('%-40s %6d %8d\n', base, numel(names), numel(t_ref));
  end
  fprintf('\n导出 %d 条\n', nok);
end

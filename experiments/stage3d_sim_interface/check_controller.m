% 检查 AGTF30e 官方模型里到底有没有 PI 控制器
%
% 要查清：
%   1. Control System 里有哪些控制律（PI / 积分器 / 增益调度）
%   2. Wf Controller 内部是什么
%   3. N1cErr 是怎么产生的（反馈闭环证据）
%   4. PLA 到 Wf 的路径

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
outDir = 'C:\pclwm\pclwm_stage3a\compare';
if ~isfolder(outDir), mkdir(outDir); end
try, close_system(MDL,0); catch, end
load_system(MDL);

fid = fopen(fullfile(outDir,'controller_structure.txt'),'w','n','UTF-8');
fprintf(fid,'=== AGTF30e 官方控制律结构 ===\n\n');

% ══ 1. 全模型搜索 PI / 积分器 / 增益 ===
fprintf(fid,'[1] 全模型的 PI / Integrator 块\n');
for bt = {'PID 1st','Integrator','DiscreteIntegrator','PID','TransferFcn'}
  blks = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                     'MatchFilter',@Simulink.match.allVariants, ...
                     'BlockType', bt{1});
  if ~isempty(blks)
    fprintf(fid,'  [%s] 共 %d 个\n', bt{1}, numel(blks));
    for i=1:min(numel(blks),10)
      fprintf(fid,'    %s\n', strrep(blks{i}, [MDL '/'], ''));
    end
  end
end

% ══ 2. Wf Controller 内部 ══
fprintf(fid,'\n[2] Wf Controller 内部结构\n');
WC = [MDL '/Control System/Wf Controller'];
if getSimulinkBlockHandle(WC) > 0
  kids = find_system(WC,'LookUnderMasks','all','FollowLinks','on', ...
                     'MatchFilter',@Simulink.match.allVariants);
  fprintf(fid,'  子块数: %d\n', numel(kids));
  for i=1:numel(kids)
    bt = get_param(kids{i},'BlockType');
    nm = strrep(kids{i}, [WC '/'], '');
    fprintf(fid,'    [%s] %s\n', bt, nm);
  end
else
  fprintf(fid,'  未找到\n');
end

% ══ 3. 整个 Control System 的顶层结构 ══
fprintf(fid,'\n[3] Control System 顶层子块\n');
CS = [MDL '/Control System'];
kids = find_system(CS,'SearchDepth',1,'LookUnderMasks','all', ...
                   'MatchFilter',@Simulink.match.allVariants);
for i=2:numel(kids)
  fprintf(fid,'  [%s] %s\n', get_param(kids{i},'BlockType'), ...
          strrep(kids{i}, [CS '/'], ''));
end

% ══ 4. N1cErr 的生成路径 ══
fprintf(fid,'\n[4] 查找 N1cErr / N1c 相关块\n');
for pat = {'N1cErr','N1cN','N1cSP','N1cRef','N1c'}
  blks = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                     'MatchFilter',@Simulink.match.allVariants, ...
                     'regexp','on','Name',pat{1});
  if ~isempty(blks)
    fprintf(fid,'  含 "%s": %d 个\n', pat{1}, numel(blks));
    for i=1:min(numel(blks),8)
      fprintf(fid,'    [%s] %s\n', get_param(blks{i},'BlockType'), ...
              strrep(blks{i}, [MDL '/'], ''));
    end
  end
end

% ══ 5. 检查 setup_Controller.m 的说明 ══
fprintf(fid,'\n[5] setup_Controller.m 中的控制律描述\n');
cf = fullfile(agtf,'SimSetup','setup_Controller.m');
if isfile(cf)
  txt = fileread(cf);
  lines = strsplit(txt, newline);
  n = 0;
  for i=1:numel(lines)
    ln = lines{i};
    if (contains(lower(ln),'pi ') || contains(lower(ln),'proportional') || ...
        contains(lower(ln),'integral') || contains(lower(ln),'ki') || ...
        contains(lower(ln),'kp')) && ~startsWith(strtrim(ln),'%')
      fprintf(fid,'  L%d: %s\n', i, strtrim(ln));
      n = n + 1;
      if n > 25, break; end
    end
  end
else
  fprintf(fid,'  文件不存在\n');
end

% ══ 6. 检查 N1cN 是否来自 PLA（静态映射）══
fprintf(fid,'\n[6] PLA -> N1cN 映射链路\n');
blks = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                   'MatchFilter',@Simulink.match.allVariants, ...
                   'regexp','on','Name','Scheduling');
for i=1:numel(blks)
  fprintf(fid,'  %s\n', strrep(blks{i}, [MDL '/'], ''));
  k2 = find_system(blks{i},'SearchDepth',1, ...
                   'MatchFilter',@Simulink.match.allVariants);
  for j=2:numel(k2)
    fprintf(fid,'    [%s] %s\n', get_param(k2{j},'BlockType'), ...
            strrep(k2{j}, [blks{i} '/'], ''));
  end
end

fclose(fid);
fprintf('PHYS_OK\n');

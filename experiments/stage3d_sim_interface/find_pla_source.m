% 确定 PLA 对应的 Model Source 编号 + 测试 Lookup 可调性
root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
outDir = 'C:\pclwm\pclwm_stage3a\step_api';
try, close_system(MDL,0); catch, end
load_system(MDL);

fprintf('=== 确定 PLA 对应的 Model Source ===\n\n');

% 检查每个 Model Source 的输出端口连到哪里
ms = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                 'regexp','on','Name','^Model Source');
% 按编号排序
nums = zeros(numel(ms),1);
for i=1:numel(ms)
  nm = get_param(ms{i},'Name');
  d = regexp(nm,'\d+','match');
  if isempty(d), nums(i)=0; else, nums(i)=str2double(d{1}); end
end
[~,idx] = sort(nums);
ms = ms(idx);

fprintf('  %-14s %-40s %-22s\n','编号','输出连线','Lookup 参数');
fprintf('  %s\n', repmat('-',1,80));

for i = 1:numel(ms)
  nm = get_param(ms{i},'Name');
  % 输出连线
  ph = get_param(ms{i},'PortHandles');
  dst = '';
  if ~isempty(ph.Outport)
    l = get_param(ph.Outport(1),'Line');
    if l > 0
      dp = get_param(l,'DstPortHandle');
      if ~isempty(dp)
        dblk = get_param(dp(1),'Parent');
        dst = get_param(dblk,'Name');
      end
    end
  end
  % 内部 Lookup
  lk = [ms{i} '/Input_profiles'];
  tbl = '';
  try
    bp = get_param(lk,'DialogParameters');
    tbl = 'Lookup';
  catch
  end
  fprintf('  %-14s %-40s %-22s\n', nm, dst, tbl);
end

% 找一个 Lookup 看参数
fprintf('\n=== Lookup 块参数结构 ===\n');
lk = [ms{1} '/Input_profiles'];
bp = get_param(lk,'DialogParameters');
fn = fieldnames(bp);
for i=1:numel(fn)
  try
    v = get_param(lk, fn{i});
    if isnumeric(v) || ischar(v) || isstring(v)
      vs = mat2str(v);
      if numel(vs)>100, vs=[vs(1:100) '...']; end
      fprintf('  %-24s = %s\n', fn{i}, vs);
    else
      fprintf('  %-24s (类型 %s)\n', fn{i}, class(v));
    end
  catch
  end
end

fclose('all');
fprintf('PHYS_OK\n');

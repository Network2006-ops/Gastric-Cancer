function classNames = organizeDataset(rawFolder, organizedFolder)
% ORGANIZEDATASET  Sort flat, prefix-named tissue images into class
% subfolders so imageDatastore can pick up folder names as labels.
%
%   Files are expected to be named like CLASS_number.png, e.g.
%   ADI_90.png, TUM_995.png. The part before the first underscore is
%   used as the class label.
%
%   classNames = organizeDataset(rawFolder, organizedFolder)

if exist(organizedFolder, 'dir')
    % Clean and rebuild so re-running main.m is idempotent
    rmdir(organizedFolder, 's');
end
mkdir(organizedFolder);

files = [dir(fullfile(rawFolder, '*.png')); ...
         dir(fullfile(rawFolder, '*.jpg')); ...
         dir(fullfile(rawFolder, '*.jpeg'))];

if isempty(files)
    error('No images found in %s. Put your tissue patch images there first.', rawFolder);
end

classNames = {};
for k = 1:numel(files)
    [~, name, ext] = fileparts(files(k).name);
    parts = strsplit(name, '_');
    label = upper(parts{1});

    classFolder = fullfile(organizedFolder, label);
    if ~exist(classFolder, 'dir')
        mkdir(classFolder);
        classNames{end+1} = label; %#ok<AGROW>
    end

    srcPath = fullfile(files(k).folder, files(k).name);
    dstPath = fullfile(classFolder, [name ext]);
    copyfile(srcPath, dstPath);
end

classNames = sort(classNames);
end

%% INITIALIZE

clear variables

restoredefaultpath
addpath /home/lau/matlab/fieldtrip
ft_defaults

%% READ HEADSHAPE

path = '/home/lau/simnibs_examples/lau/mri_convert/fs_lau_nocleanup/bem';

filenames = {'csf.stl' 'skull.stl' 'skin.stl'};
headshapes = cell(1, 3);

counter = 1;
for filename = filenames
    headshapes{counter} = ft_read_headshape(fullfile(path, filename));
    counter = counter + 1;
end

%% CREATE MESH

meshes = cell(1, 3);
counter = 1;
for headshape = headshapes

    cfg = [];
    cfg.method = 'headshape';
    cfg.headshape = headshape{1};
    
    meshes{counter} = ft_prepare_mesh(cfg);
    counter = counter + 1;
end

meshes{1}.tissue = 'brain';
meshes{2}.tissue = 'skull';
meshes{3}.tissue = 'scalp';

bnd = [meshes{:}];

%% PLOT MESHES

figure;
hold on

ft_plot_mesh(meshes{1}, 'facealpha', 1, 'facecolor', 'brain')
ft_plot_mesh(meshes{2}, 'facealpha', 0.2, 'facecolor', 'black')
ft_plot_mesh(meshes{3}, 'facealpha', 0.2, 'facecolor', 'skin')
% 
%% HEAD MODEL

cfg = [];
cfg.method = 'openmeeg';
cfg.conductivity = [0.3 0.006 0.3];

headmodel = ft_prepare_headmodel(cfg, bnd);

% cfg = [];
% cfg.method = 'singleshell';
% 
% headmodel = ft_prepare_headmodel(cfg, meshes{1});
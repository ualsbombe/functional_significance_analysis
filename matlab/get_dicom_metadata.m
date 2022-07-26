path = '/home/lau/projects/functional_cerebellum/raw/0009/20210812_141341/MR/';

scans = {'005.t1_mprage_3D_sag_fatsat' '007.t2_tse_sag_HighBW'};
parameters = {'Height' 'Width' 'SliceThickness' 'PixelBandwidth' ...
              'FlipAngle' 'InversionTime' 'EchoTime' ...
               'RepetitionTime'};



for scan_index = 1:length(scans)
    scan = scans{scan_index};
    full_folder = fullfile(path, scan, 'files');
    cd(full_folder)
    files = dir();
    first_file = files(3).name;
    info = dicominfo(fullfile(full_folder, first_file));
    for parameter_index = 1:length(parameters)
        parameter = parameters{parameter_index};
        scan_type = scan(5:6);
        if ~(strcmp(scan_type, 't2') && strcmp(parameter, 'InversionTime'))
            disp([scan_type ' ' parameter])
            disp(info.(parameter))
        end
        
    end
    disp([scan_type ' n files: ' num2str(size(files, 1) - 2)])
end
        
        
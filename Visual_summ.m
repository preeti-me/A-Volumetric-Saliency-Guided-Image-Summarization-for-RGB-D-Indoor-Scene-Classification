
imgRoot = 'data\RGB\';
RGBSalRoot = 'data\mask\';

%vs = './vs/';
%mkdir(vs);

rgbFiles = dir(fullfile(imgRoot, '*.png'));
salFiles = dir(fullfile(RGBSalRoot, '*.png'));

%%
for ii=1:length(rgbFiles)
    disp(ii);
    
    imgName = rgbFiles(ii).name;  % load RGB data
    
    I =(imread(fullfile(imgRoot, imgName)));
  
    salName = salFiles(ii).name;   % load RGB saliency data

     S=(imread(fullfile(RGBSalRoot, salName)));
    S=repmat(S,[1 1 3]);
    %VS=I.*S;

   VS = I;
   VS(S == 0) = 0;


   imshow(VS); title('Saliency-guided image summary');

end
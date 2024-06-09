clc; clear all;
addpath('rgbd/')
%addpath('data/')
addpath('C:/Users/Vision Lab/Desktop/rgb-d/code/code/')


Database=load('nyu_depth_v2_labeled.mat');
rgbimg=Database.images;  %%rgb images
depimg=  Database.depths;
 
Image= 340;

rgbImg=(rgbimg(:,:,:,Image)); 
depImg=(depimg(:,:,Image));

rgbImg=rgbImg(21:470, 11:630, :); depImg=depImg(21:470, 11:630, :);

depImg=(mat2gray(depImg));


%%
 scD=2; sc=2;
rgbImg = rgbImg(1:scD:end, 1:scD:end, :);
szh(1) = uint16(size(rgbImg(1:sc:end,1:sc:end,:),1));
szh(2) = uint16(size(rgbImg(1:sc:end,1:sc:end,:),2));

% Convert to 3D points
[info3D, ~] = DepthtoCloud(depImg);
info3Dsc = info3D(1:scD:end, 1:scD:end, :);
[r,c,d] = size(info3Dsc);
featVec3D = normalizeandscale(reshape(info3Dsc, r*c, d));

%%
lc=load(['lc' num2str(Image) '.mat']); %%load the segmentated labels
lc=lc.lc;

BWlc1 = boundarymask(lc);  %figure;imshow(BWlc1);

%%
imgLabels1 = assignRandomLabel(BWlc1);

 are=regionprops( imgLabels1,'Area');

  propied=regionprops( imgLabels1,'BoundingBox');
%  %propiedall=regionprops( imgLabels1,'all');
 pxidx=regionprops( imgLabels1,'PixelIdxList');


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('volume computation: running ... \n');
%vi=1;

un=unique(lc);
for j=1:length(un)
  ti=  un(j);

  inx = find(lc==ti);
 data=featVec3D (inx, :); 
  [TriIdx, V] = convhull(data(:,1),data(:,2),data(:,3));
  vol(j)=V;
 % vi=vi+1;
%figure; trisurf(TriIdx, data(:,1),data(:,2),data(:,3))
xd=max(max(data(:,1)))-min(min(data(:,1)));
xd=max(max(data(:,2)))-min(min(data(:,2)));
xd=max(max(data(:,1)))-min(min(data(:,1)));
yd=max(max(data(:,2)))-min(min(data(:,2)));
zd=max(max(data(:,3)))-min(min(data(:,3)));
volume=xd*yd*zd;
volumefinal(j)=volume;
end

comSal =  vol;
%comSal =  volumefinal;
spnumnew = length(unique(lc)); 
%%
regions = calculateRegionProps(spnumnew,lc);


 %%
%color =rgbImg;

blobMeasurements = regionprops(lc, 'all');   % Get all the blob properties.
numberOfBlobs = size(blobMeasurements, 1);

segmented_images = cell(1,max(max(lc)));
rgb_label = repmat(lc,[1 1 3]);

for k = 1 :numberOfBlobs
    vo=volumefinal(k);

if vo>=0.2  %max(volumefinal)/80 
    
    color =ones(225,310,3); 
    %color =rgbImg;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color; 

    segmented_new_images= segmented_images{k};

end  
end


figure; imshow(segmented_new_images);



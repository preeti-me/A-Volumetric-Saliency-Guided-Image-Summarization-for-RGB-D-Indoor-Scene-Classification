clear all; clc;
addpath('rgbd/')
addpath('data/')

 Database=load('nyu_depth_v2_labeled.mat');
 rgbimg=Database.images;  %%rgb images
 Label_gndTru=Database.labels; %groundtruth segments

for Image=340

rgbImg=(rgbimg(:,:,:,Image));
rgbImg=rgbImg(21:470, 11:630, :);


L_groundTruth=(Label_gndTru(:,:,Image));  L_groundTruth=L_groundTruth(21:470, 11:630, :);

scD=2;
szh(1) = uint16(size(rgbImg(1:scD:end,1:scD:end,:),1));
szh(2) = uint16(size(rgbImg(1:scD:end,1:scD:end,:),2));

%%
 rgbImg=(rgbImg(1:scD:end, 1:scD:end, :));
 I=rgbImg;

lc=load(['lc' num2str(Image) '.mat']); %%load the segmentated labels
lc=lc.lc;  
%imagesc(lc);

segmented_images = cell(1,max(max(lc)));
rgb_label = repmat(lc,[1 1 3]);
for k = 1:max(max(lc)) %L-label matrix
    color = I;  %im-Color img
    
   color(rgb_label ~= k) = 0;
    segmented_images{k} = color; 

   segmented_new_images= segmented_images{k};
   figure; imshow(segmented_new_images);


end
end




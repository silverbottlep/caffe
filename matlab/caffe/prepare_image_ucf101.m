% ------------------------------------------------------------------------
function images = prepare_image_ucf101(im)
% ------------------------------------------------------------------------
CROPPED_DIM = 224;
im_height = size(im,1);
im_width = size(im,2);

% oversample (4 corners, center, and their x-axis flips)
% images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices_y = [0 im_height-CROPPED_DIM] + 1;
indices_x = [0 im_width-CROPPED_DIM] + 1;
curr = 1;
for i = indices_y
  for j = indices_x
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center_y = floor(indices_y(2) / 2)+1;
center_x = floor(indices_x(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);

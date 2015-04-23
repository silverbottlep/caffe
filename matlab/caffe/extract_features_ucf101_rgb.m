src_dir = '../../data/ucf101/ucf101_image/';
out_dir = '../../data/ucf101/ucf101_rgb_feat/';
list_file = '../../data/ucf101/train1.txt';

model_def_file = '../../examples/twostream/spatialnet_ft_deploy_conv5.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_45000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 0);

image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));

list_fid = fopen(list_file);
line = fgetl(list_fid);
num_item = 0;
while ischar(line)
	num_item = num_item + 1;
	itemlist{num_item} = textscan(line,'%s %d %d\n');
	item_name{num_item} = itemlist{num_item}{1};
	item_label(num_item) = itemlist{num_item}{2};
	nframes(num_item) = itemlist{num_item}{3};
	line = fgetl(list_fid);
end
fclose(list_fid);

accuracy = 0;
CROPPED_DIM = 224;
input = zeros(256, 340, 3, 1, 'single');
for i=1:num_item
	item = char(item_name{i});
	feat = [];
	for j=1:nframes(i)
		filename = strcat(src_dir, item, '/', item, '_f', ...
			num2str(j,'%04u'), '.jpg');
		
		im = imread(char(filename));
		im = single(im);
		if size(im,3) == 1
				im = cat(3,im,im,im);
		end
		im = im(:,:,[3 2 1]) - IMAGE_MEAN;
		input = permute(im,[2 1 3]);
		output_data = caffe('forward', {input});
		conv5 = permute(output_data{1}, [2 1 3]);
		conv5 = reshape(conv5,[size(conv5,1) size(conv5,2)*size(conv5,3)]);
		%feat(:,:,:,j) = single(conv5);
		feat = [feat; conv5];
	end
	feat = sparse(double(feat));

%	feat_rgb_conv5(i).feat = feat;
%	feat_rgb_conv5(i).item = item;
%	feat_rgb_conv5(i).nframes = nframes(i);
%	feat_rgb_conv5(i).label = item_label(i);

	fprintf('processing %s(%d/%d) num_frame: %d\n', item, i, num_item, nframes(i) );
	outfile = [out_dir item '.mat'];
	save(outfile, 'feat','-v7.3');
%	if mode(i,1000) == 0
%		save('features_rgb_conv5.mat','feat_rgb_conv5');
%	end
end

%save -v7.3 'temp.mat' 'feat';
%hdf5write('temp.h5','./',feat);
%save('features_rgb_conv5.mat','feat_rgb_conv5');

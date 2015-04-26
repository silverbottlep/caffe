src_dir = '../../data/ucf101/ucf101_image/';
list_file = '../../data/ucf101/all.txt';

%model_def_file = '../../examples/twostream/spatialnet_ft_deploy_conv5.prototxt';
%model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_45000.caffemodel';
%out_dir = '../../data/ucf101/ucf101_rgb_feat/';
%height = 256; width = 340;

model_def_file = '../../examples/twostream/spatialnet_vgg19_ft_deploy_conv5.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_vgg19.caffemodel';
out_dir = '../../data/ucf101/ucf101_rgb_vgg19_feat/';
height = 240; width = 324;

use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 2);

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
input = zeros(height, width, 3, 1, 'single');
for i=1:num_item
	item = char(item_name{i});
	feat = [];
	database_name = [out_dir item '.h5'];
	for j=1:nframes(i)
		filename = strcat(src_dir, item, '/', item, '_f', ...
			num2str(j,'%04u'), '.jpg');
		
		im = imread(char(filename));
		im = single(im);
		if size(im,3) == 1
				im = cat(3,im,im,im);
		end
		im = im(:,:,[3 2 1]) - IMAGE_MEAN;
		im = imresize(im,[height width]);
		input = permute(im,[2 1 3]);
		output_data = caffe('forward', {input});
		conv5 = single(permute(output_data{1}, [2 1 3]));
		%conv5 = reshape(conv5,[size(conv5,1)*size(conv5,3) size(conv5,2)]);
		%feat = [feat conv5];

		h5create(database_name, ['/' num2str(sprintf('%04d',j))], size(conv5), ...
							'Datatype', 'single','ChunkSize',size(conv5),'Deflate', 1);
		h5write(database_name,['/' num2str(sprintf('%04d',j))], conv5);
	end

	fprintf('processing %s(%d/%d) num_frame: %d\n', item, i, num_item, nframes(i) );
end

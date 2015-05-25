src_dir = '../../data/ucf101/ucf101_image/';
list_file = '../../data/ucf101/test3.txt';

%model_def_file = '../../examples/twostream/spatialnet_ft_deploy.prototxt';
%model_file = '../../examples/split3/snapshot/spatialnet_ft_iter_45000.caffemodel';
%output_name = 'rgb_result_split3.mat';
%batch = 1;

% for vgg19, different layer name( fc6 : fc6_ucf101 ), so we need to have different deploy file
% because of pool5 feature map size( 6x6, 7x7)
%model_def_file = '../../examples/twostream/spatialnet_vgg19_ft_deploy.prototxt';
model_def_file = '../../examples/split3/spatialnet_vgg19_ft_deploy.prototxt';
model_file = '../../examples/split3/snapshot/spatialnet_vgg19_ft_iter_60000.caffemodel';
output_name = 'rgb_vgg19_result_split3.mat';
batch = 5;

use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file);
caffe('set_device',0);

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
nsamples = 25;
input = zeros(CROPPED_DIM, CROPPED_DIM, 3, (nsamples/batch)*10, 'single');
for i=1:num_item
	scores = [];
	item = char(item_name{i});

	gap = floor(single(nframes(i)) / single(nsamples));
	for b=1:batch
		for j=1:nsamples/batch
			frame_num = ((b-1)*(nsamples/batch)+j-1)*gap+1;
			filename = strcat(src_dir, item, '/', item, '_f', ...
				num2str(frame_num,'%04u'), '.jpg');
			
			im = imread(char(filename));
			im = single(im);
			if size(im,3) == 1
					im = cat(3,im,im,im);
			end
			im = im(:,:,[3 2 1]) - IMAGE_MEAN;
			images{j} = prepare_image_ucf101(im);
		end
		for j=1:nsamples/batch
			input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
		end
		output_data = caffe('forward', {input});
		o = squeeze(output_data{1});
		scores = [scores o];
	end
	s = sum(scores,2)/size(scores,2);
	[max_s, idx] = max(s);
	output_label(i) = idx-1;
	output_score(:,i) = s;

	if item_label(i) == output_label(i)
		accuracy = accuracy+1;
	end

	rgb_result(i).feat = scores;
	rgb_result(i).item_name = item;
	rgb_result(i).nframes = nframes(i);
	rgb_result(i).label = item_label(i);
	rgb_result(i).output_label = output_label(i);

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
end

save(output_name, 'rgb_result');
fprintf('total accuracy:%s\n',accuracy/num_item);

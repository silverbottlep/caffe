src_dir = '../../data/ucf101/ucf101_image/';
flow_dir = '../../data/ucf101/ucf101_optmag/';
list_file = '../../data/ucf101/test1.txt';

% get full model weight
%model_def_file = '../../examples/twostream/spatialnet_ft_consilience_deploy.prototxt';
%model_file = '../../examples/twostream/snapshot/spatialnet_ft_consilience_B_1_iter_40000.caffemodel';
%output_name = 'rgb_consilience_result.mat';
%batch = 1;
%FLOW_DIM = 13;

model_def_file = '../../examples/twostream/spatialnet_vgg19_ft_consilience_deploy.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_vgg19_ft_consilience_B_3_iter_60000.caffemodel';
output_name = 'rgb_vgg19_consilience_result.mat';
batch = 5;
FLOW_DIM = 14;

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
CROPPED_DIM = 224;
nsamples = 25;
input = zeros(CROPPED_DIM, CROPPED_DIM, 3, (nsamples/batch)*10, 'single');
flow_input = zeros(FLOW_DIM, FLOW_DIM, 1, (nsamples/batch)*10, 'single');
for i=1:num_item
	scores = [];
	item = char(item_name{i});

	gap = floor(single(nframes(i)) / single(nsamples));
	for b=1:batch
		for j=1:nsamples/batch
			frame_num = ((b-1)*(nsamples/batch)+j-1)*gap+1;
			filename = strcat(src_dir, item, '/', item, '_f', ...
				num2str(frame_num,'%04u'), '.jpg');
			flow_filename = strcat(flow_dir, item, '/', item, '_f', ...
				num2str(frame_num,'%04u'), '_optmag.jpg');
			
			im = imread(char(filename));
			im = single(im);
			if size(im,3) == 1
					im = cat(3,im,im,im);
			end
			im = im(:,:,[3 2 1]) - IMAGE_MEAN;
			images{j} = prepare_image_ucf101(im);

			flow = imread(char(flow_filename));
			flow = single(flow);
			flows{j} = prepare_image_ucf101(flow);
			%images{j} = prepare_image_ucf101(im);
			flows{j} = imresize(flows{j},[FLOW_DIM FLOW_DIM]);
		end
		for j=1:nsamples/batch
			input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
		end
		for j=1:nsamples/batch
			flow_input(:,:,:,(j-1)*10+1:(j-1)*10+10) = flows{j};
		end
		output_data = caffe('forward', {input; flow_input});
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

	rgb_consilience_result(i).feat = scores;
	rgb_consilience_result(i).item_name = item;
	rgb_consilience_result(i).nframes = nframes(i);
	rgb_consilience_result(i).label = item_label(i);
	rgb_consilience_result(i).output_label = output_label(i);

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
end

save(output_name, 'rgb_consilience_result');
fprintf('total accuracy:%s\n',accuracy/num_item);

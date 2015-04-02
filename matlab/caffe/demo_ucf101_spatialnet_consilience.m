src_dir = '../../data/ucf101/ucf101_image/';
flow_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/test1.txt';

% get full model weight
model_def_file = '../../examples/twostream/spatialnet_ft_deploy.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_34000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 2);
model = caffe('get_weights');
W1 = model(6).weights{1}; W1 = W1';
b1 = model(6).weights{2};
W2 = model(7).weights{1}; W2 = W2';
b2 = model(7).weights{2};
W3 = model(8).weights{1}; W3 = W3';
b3 = model(8).weights{2};
caffe('reset');

% model for pool5
model_def_file = '../../examples/twostream/spatialnet_ft_deploy_pool5.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_34000.caffemodel';
matcaffe_init(use_gpu, model_def_file, model_file, 2);

image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));
flow_mean = imread('../../data/ucf101/ucf101_flow_mean.binaryproto.jpg');
FLOW_MEAN = single(flow_mean);

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
input = zeros(CROPPED_DIM, CROPPED_DIM, 3, nsamples*10, 'single');
%for i=1:num_item
for i=1:1
	item = char(item_name{i});

	gap = floor(single(nframes(i)) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
		filename = strcat(src_dir, item, '/', item, '_f', ...
			num2str(frame_num,'%04u'), '.jpg');
		flowname = strcat(flow_dir, item, '/', item, '_f', ...
			num2str(frame_num,'%04u'), '_opt.jpg');
		
		im = imread(char(filename));
		im = single(im);
		if size(im,3) == 1
				im = cat(3,im,im,im);
		end
		im = im(:,:,[3 2 1]) - IMAGE_MEAN;
		images{j} = prepare_image_ucf101(im);

		flow = imread(char(flowname));
		flow = single(flow);
		flow = flow - FLOW_MEAN;

	end
	for j=1:nsamples
		input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
	end
	output_data = caffe('forward', {input});

	pool5 = output_data{1};
	feat = reshape(pool5, [size(pool5,1)*size(pool5,2)*size(pool5,3) size(pool5,4)]);
	os1 = W1*feat + repmat(b1,[1 nsamples*10]);
	os1 = max(0,os1);
	os2 = W2*os1 + repmat(b2,[1 nsamples*10]);
	os2 = max(0,os2);
	os3 = W3*os2 + repmat(b3,[1 nsamples*10]);

	% softmax
	os3 = os3 - repmat(max(os3,[],1),[size(os3,1) 1]);
	os3 = exp(os3);
	os3 = os3./repmat(sum(os3),[size(os3,1) 1]);
	softmax = os3;

	s = sum(softmax,2)/size(softmax,2);
	[max_s, idx] = max(s);
	output_label(i) = idx-1;
	output_score(:,i) = s;

	if item_label(i) == output_label(i)
		accuracy = accuracy+1;
	end

	rgb_consilience_result(i).softmax = softmax;
	rgb_consilience_result(i).item_name = item;
	rgb_consilience_result(i).nframes = nframes(i);
	rgb_consilience_result(i).label = item_label(i);
	rgb_consilience_result(i).output_label = output_label(i);

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
end
caffe('reset');
save('rgb_consilience_result.mat', 'rgb_consilience_result');
fprintf('total accuracy:%s\n',accuracy/num_item);

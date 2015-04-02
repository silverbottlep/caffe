src_dir = '../../data/ucf101/ucf101_image/';
flow_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/test1.txt';
minmax_file = '../../data/ucf101/minmax.csv';

% get full model weight
model_def_file = '../../examples/twostream/spatialnet_ft_deploy.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_34000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 0);
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
matcaffe_init(use_gpu, model_def_file, model_file, 0);

image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));
%flow_mean = imread('../../data/ucf101/ucf101_flow_mean.binaryproto.jpg');
%FLOW_MEAN = single(flow_mean);

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
minmax = csvread(minmax_file);
image_list = dir(src_dir);
image_list = image_list(3:end);
keyset = {image_list(:).name};
valueset = 1:length(image_list);   
name_to_idx = containers.Map(keyset,valueset);
num_scale = 7;

accuracy = zeros(num_scale+1,1);
CROPPED_DIM = 224;
nsamples = 25;
input = zeros(CROPPED_DIM, CROPPED_DIM, 3, nsamples*10, 'single');
flow_pool5 = zeros(6, 6, 512, nsamples*10, 'single');
for i=1:num_item
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
		minmax_idx = name_to_idx(item);
		minimum = minmax(minmax_idx,1);
		maximum = minmax(minmax_idx,2);
		scale = 255/(maximum - minimum);
		rescaled_flow = minimum + flow/scale;
		flow_x = rescaled_flow(1:size(flow,1)/2,:);
		flow_y = rescaled_flow(size(flow,1)/2+1:end,:);
		flow_mag = sqrt((flow_x.^2 + flow_y.^2));
		flow_mag = prepare_flow_ucf101(flow_mag);
		flow_mag = imresize(flow_mag,[6 6]);
		flow_input{j}{1} = flow_mag;
		flow_input{j}{2} = scale_data(flow_mag,0,1);
		flow_input{j}{3} = scale_data(flow_mag,0,2);
		flow_input{j}{4} = scale_data(flow_mag,0,4);
		flow_input{j}{5} = scale_data(flow_mag,1,2);
		flow_input{j}{6} = scale_data(flow_mag,1,4);
		flow_input{j}{7} = scale_data(flow_mag,1,8);
	end
	for j=1:nsamples
		input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
	end
	output_data = caffe('forward', {input});

	pool5 = output_data{1};
	for sc=1:num_scale+1
		if sc == num_scale+1
			new_pool5 = pool5;
		else
			for j=1:nsamples
				ff = flow_input{j}{sc};
				for k=1:10
					flow_pool5(:,:,:,(j-1)*10+k) = repmat(ff(:,:,k),1,1,size(pool5,3));
				end
			end
			new_pool5 = pool5.*flow_pool5;
		end
		feat = reshape(new_pool5, [size(new_pool5,1)*size(new_pool5,2)*size(new_pool5,3) size(new_pool5,4)]);
		os1 = W1*feat + repmat(b1,[1 nsamples*10]);
		os1 = max(0,os1);
		os2 = W2*os1 + repmat(b2,[1 nsamples*10]);
		os2 = max(0,os2);
		os3 = W3*os2 + repmat(b3,[1 nsamples*10]);

		% softmax
		os3 = os3 - repmat(max(os3,[],1),[size(os3,1) 1]);
		os3 = exp(os3);
		os3 = os3./repmat(sum(os3),[size(os3,1) 1]);
		soft_max{sc} = os3;

		s = sum(soft_max{sc},2)/size(soft_max{sc},2);
		[max_s(sc), idx] = max(s);
		output_label(sc) = idx-1;

		if item_label(i) == output_label(sc)
			accuracy(sc) = accuracy(sc)+1;
		end

	end

	rgb_consilience_result(i).soft_max = soft_max;
	rgb_consilience_result(i).item_name = item;
	rgb_consilience_result(i).nframes = nframes(i);
	rgb_consilience_result(i).label = item_label(i);
	rgb_consilience_result(i).output_label = output_label;

	for sc=1:num_scale+1
		fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
				item, i, num_item, (accuracy(sc)/i)*100, output_label(sc), max_s(sc), item_label(i));
	end
end
caffe('reset');
save('rgb_consilience_result.mat', 'rgb_consilience_result');
fprintf('total accuracy:%s\n',accuracy/num_item);

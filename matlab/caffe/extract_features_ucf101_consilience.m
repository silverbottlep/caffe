rgb_dir = '../../data/ucf101/ucf101_image/';
flow_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/train1.txt';
%list_file = '../../data/ucf101/test1.txt';

%model_file = '../../examples/consilience/snapshot/consilience_nonorm_d2048_drop7_iter_60000.caffemodel';
%model_file = '../../examples/consilience/snapshot/consilience_nonorm_d2048_drop7_bias_iter_40000.caffemodel';
model_def_file = '../../examples/cons/fusion_triple_deploy.prototxt';
model_file = '../../examples/cons/snapshot/fusion_triple.caffemodel';

use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file);
caffe('set_device',0);

flow_mean = imread('../../data/ucf101/ucf101_flow_mean.binaryproto.jpg');
FLOW_MEAN = single(flow_mean);
image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));

list_fid = fopen(list_file);
line = fgetl(list_fid);
num_item = 0;
while ischar(line)
	num_item = num_item + 1;
	itemlist{num_item} = textscan(line,'%s %d %d %f %f\n');
	item_name{num_item} = itemlist{num_item}{1};
	item_label(num_item) = itemlist{num_item}{2};
	nframes(num_item) = itemlist{num_item}{3};
	line = fgetl(list_fid);
end
fclose(list_fid);

rgb_accuracy = 0;
flow_accuracy = 0;
cons_accuracy = 0;
triple_accuracy = 0;
CROPPED_DIM = 224;
nsamples = 25;
nchannels = 10;
ncrops = 10;
rgb_input = zeros(CROPPED_DIM, CROPPED_DIM, 3, ncrops, 'single');
flow_input = zeros(CROPPED_DIM, CROPPED_DIM, nchannels*2, ncrops, 'single');
scores = zeros(101,nsamples*ncrops,3);

for i=1:num_item
	item = char(item_name{i});
	tic;
	gap = floor(single(nframes(i)-nchannels) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
	
		flow = [];
		for k=1:nchannels
			flow_filename = strcat(flow_dir, item, '/', item, '_f', ...
				num2str(frame_num+k-1,'%04u'), '_opt.jpg');
			im = imread(char(flow_filename));
			im = single(im);
			im = im - FLOW_MEAN;
			flow_x = im(1:size(im,1)/2,:);
			flow_y = im(size(im,1)/2+1:end,:);
			flow(:,:,(k-1)*2+1) = flow_x;
			flow(:,:,(k-1)*2+2) = flow_y;

			if k==(nchannels/2)
				rgb_filename = strcat(rgb_dir, item, '/', item, '_f', ...
				num2str(frame_num+k,'%04u'), '.jpg');
				im = imread(char(rgb_filename));
				im = single(im);
				if size(im,3) == 1
						im = cat(3,im,im,im);
				end
				im = im(:,:,[3 2 1]) - IMAGE_MEAN;
				rgb_input = prepare_image_ucf101(im);
			end
		end
		flow_input = prepare_image_ucf101(flow);

		output_data = caffe('forward', {single(rgb_input); single(flow_input)} );
		scores(:,(j-1)*ncrops+1:(j-1)*ncrops+ncrops,1) = squeeze(output_data{1});
		scores(:,(j-1)*ncrops+1:(j-1)*ncrops+ncrops,2) = squeeze(output_data{2});
		scores(:,(j-1)*ncrops+1:(j-1)*ncrops+ncrops,3) = squeeze(output_data{3});
	end

	features = squeeze(sum(scores,2)/size(scores,2));
	[triple_max_s, idx] = max(sum(features,2)/size(features,2));
	triple_label = idx-1;
	if item_label(i) == triple_label
		triple_accuracy = triple_accuracy + 1;
	end

	rgb_feat(:,i) = features(:,1);
	flow_feat(:,i) = features(:,2);
	cons_feat(:,i) = features(:,3);
	[rgb_max_s, idx] = max(rgb_feat(:,i));
	rgb_label = idx-1;
	if item_label(i) == rgb_label
		rgb_accuracy = rgb_accuracy + 1;
	end
	[flow_max_s, idx] = max(flow_feat(:,i));
	flow_label = idx-1;
	if item_label(i) == flow_label
		flow_accuracy = flow_accuracy + 1;
	end
	[cons_max_s, idx] = max(cons_feat(:,i));
	cons_label = idx-1;
	if item_label(i) == cons_label
		cons_accuracy = cons_accuracy + 1;
	end

	fprintf('processing %s(%d/%d) gt:%d\n', item, i, num_item, item_label(i));
	fprintf('rgb: %d, %.2f, (accuracy:%.2f)\n', rgb_label, rgb_max_s, 100*rgb_accuracy/i);
	fprintf('flow: %d, %.2f, (accuracy:%.2f)\n', flow_label, flow_max_s, 100*flow_accuracy/i);
	fprintf('cons: %d, %.2f, (accuracy:%.2f)\n', cons_label, cons_max_s, 100*cons_accuracy/i);
	fprintf('triple: %d, %.2f, (accuracy:%.2f)\n', triple_label, triple_max_s, 100*triple_accuracy/i);
	toc;
end

train.rgb_feat = rgb_feat;
train.flow_feat = flow_feat;
train.cons_feat = cons_feat;
train.label = item_label;
train.item_name = item_name;
save('train_split1.mat', 'train');

%test.rgb_feat = rgb_feat;
%test.flow_feat = flow_feat;
%test.cons_feat = cons_feat;
%test.label = item_label;
%test.item_name = item_name;
%save('test_split1.mat', 'test');

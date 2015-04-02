src_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/train1.txt';

model_def_file = '../../examples/twostream/temporalnet_deploy_pool5.prototxt';
model_file = '../../examples/twostream/snapshot/temporalnet_iter_200000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 2);

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
IMAGE_DIM = 256;
nsamples = 25;
nchannels = 10;
input = zeros(CROPPED_DIM, CROPPED_DIM, nchannels*2, nsamples*10, 'single');

for i=1:num_item
	item = char(item_name{i});

	gap = floor(single(nframes(i)-nchannels) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
	
		flow = [];
		for k=1:nchannels
			filename = strcat(src_dir, item, '/', item, '_f', ...
				num2str(frame_num+k-1,'%04u'), '_opt.jpg');
			im = imread(char(filename));
			im = single(im);
			im = im - FLOW_MEAN;
			flow_x = im(1:size(im,1)/2,:);
			flow_y = im(size(im,1)/2+1:end,:);
			flow(:,:,(k-1)*2+1) = flow_x;
			flow(:,:,(k-1)*2+2) = flow_y;

		end
%		center_x = floor( (size(flow,2)-IMAGE_DIM+1) / 2)+1;
%		flow = flow(:,center_x:center_x+IMAGE_DIM-1,:);
		images{j} = prepare_image_ucf101(flow);

%		temp = prepare_image_ucf101(flow);
%		images(:,:,:,(j-1)*10+1:(j-1)*10+10) = temp;
%		for k=1:10
%			subplot(3,4,k); imagesc(temp(:,:,(k-1)*2+1,5)); colorbar;
%		end
%		pause;
	end
	
	for j=1:nsamples
		input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
	end

	output_data = caffe('forward', {input});
	
	feat_flow_pool5(i).feat = output_data{1};
	feat_flow_pool5(i).label = item_label(i);

	fprintf('processing %s(%d/%d)\n', item, i, num_item);
end

save('features_flow_pool5.mat', 'feat_flow_pool5');
fprintf('total accuracy:%s\n',accuracy/num_item);

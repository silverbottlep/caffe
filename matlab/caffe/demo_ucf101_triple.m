load('rgb_result.mat');
%load('rgb_vgg19_result.mat');
load('flow_result.mat');
load('consilience_result.mat');

accuracy = 0;
rgb_accuracy = 0;
flow_accuracy = 0;
consilience_accuracy = 0;

total_item = length(consilience_result);
for i=1:total_item
	label = consilience_result(i).label;

	rgb_feat = rgb_result(i).feat;
	rgb_feat = sum(rgb_feat,2)/size(rgb_feat,2);
	[max_s, idx] = max(rgb_feat);
	output_label = idx-1;
	if label == output_label
		rgb_accuracy = rgb_accuracy + 1;
	end

	flow_feat = flow_result(i).feat;
	flow_feat = sum(flow_feat,2)/size(flow_feat,2);
	[max_s, idx] = max(flow_feat);
	output_label = idx-1;
	if label == output_label
		flow_accuracy = flow_accuracy + 1;
	end

	consilience_feat = consilience_result(i).feat;
	consilience_feat = sum(consilience_feat,2)/size(consilience_feat,2);
	[max_s, idx] = max(consilience_feat);
	output_label = idx-1;
	if label == output_label
		consilience_accuracy = consilience_accuracy + 1;
	end

	feat = rgb_feat.*0.2 + flow_feat.*0.3 + consilience_feat.*0.5;
	[max_s, idx] = max(feat);
	output_label = idx-1;
	if label == output_label
		accuracy = accuracy+1;
	end
	fprintf('processing %s \n',consilience_result(i).item_name);
end

fprintf('rgb accuracy = %f\n', rgb_accuracy/total_item);
fprintf('flow accuracy = %f\n', flow_accuracy/total_item);
fprintf('consilience accuracy = %f\n', consilience_accuracy/total_item);
fprintf('accuracy(0.3:0.3:0.4) = %f\n', accuracy/length(consilience_result));

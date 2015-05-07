%load('rgb_result.mat');
%load('consilience_result.mat');
load('flow_result.mat');
load('rgb_vgg19_result.mat');
load('vgg19_consilience_result.mat');

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
	rgb_output_label = idx-1;
	if label == rgb_output_label
		rgb_accuracy = rgb_accuracy + 1;
	end

	flow_feat = flow_result(i).feat;
	flow_feat = sum(flow_feat,2)/size(flow_feat,2);
	[max_s, idx] = max(flow_feat);
	flow_output_label = idx-1;
	if label == flow_output_label
		flow_accuracy = flow_accuracy + 1;
	end

	consilience_feat = consilience_result(i).feat;
	consilience_feat = sum(consilience_feat,2)/size(consilience_feat,2);
	[max_s, idx] = max(consilience_feat);
	consilience_output_label = idx-1;
	if label == consilience_output_label
		consilience_accuracy = consilience_accuracy + 1;
	end

	feat = rgb_feat.*2.0 + flow_feat.*2.0 + consilience_feat.*1.0;
	[max_s, idx] = max(feat);
	triple_output_label = idx-1;
	if label == triple_output_label
		accuracy = accuracy+1;
	end
	
	fprintf('processing %s gt:%d (rgb %d, %.2f) (flow %d, %.2f) (consilience %d, %.2f) (triple %d, %.2f)\n', ...
					consilience_result(i).item_name, label, rgb_output_label, 100*rgb_accuracy/i, flow_output_label, ...
					100*flow_accuracy/i, consilience_output_label, 100*consilience_accuracy/i, triple_output_label, 100*accuracy/i);
end

fprintf('rgb accuracy = %f\n', rgb_accuracy/total_item);
fprintf('flow accuracy = %f\n', flow_accuracy/total_item);
fprintf('consilience accuracy = %f\n', consilience_accuracy/total_item);
fprintf('accuracy(1:2:3) = %f\n', accuracy/length(consilience_result));

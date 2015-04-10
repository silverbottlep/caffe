load('rgb_consilience_result.mat');
load('rgb_result.mat');
load('flow_result.mat');

alpha = 0.1:0.1:0.9;
accuracy = zeros(length(alpha),1);
rgb_accuracy = 0;
flow_accuracy = 0;
total_item = length(rgb_consilience_result);
for i=1:total_item
	label = rgb_consilience_result(i).label;
	rgb_feat = rgb_consilience_result(i).feat;
	%rgb_feat = rgb_result(i).feat;
	
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

	for j=1:length(alpha)
		feat = rgb_feat.*alpha(j) + flow_feat.*(1-alpha(j));
		[max_s, idx] = max(feat);
		output_label = idx-1;
		if label == output_label
			accuracy(j) = accuracy(j)+1;
		end
	end
	fprintf('processing %s \n',rgb_consilience_result(i).item_name);
end

fprintf('rgb accuracy = %f\n', rgb_accuracy/total_item);
fprintf('flow accuracy = %f\n', flow_accuracy/total_item);
for i=1:length(alpha)
	fprintf('accuracy(alpha:%f) = %f\n',alpha(i), accuracy(i)/length(rgb_consilience_result));
end

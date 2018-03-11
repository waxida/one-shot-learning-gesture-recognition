Attention：
	1.The original Tensorlayer does not support the convolutional LSTM, 
      so the tensorlayer/layers.py needs to be replaced with tensorlayer-layers.py. 
	2.Tensorflow-0.11 
	3.Prepare the data
		(1)Convert each video files into images using extract_frames.sh in the dataset_splits/video2image.tar.gz. 
		  Before running extract_frames.sh, you should change the ROOTDIR in extract_frames.sh, 
		  so that IsoGD_phase_1 and IsoGD_phase_2 do exist under $ROOTDIR.
		(2)Replace the path "/ssd/dataset" in the files under "dataset_splits" with the path "$ROOTDIR"
	4.Training Stage
		Use training_*.py to finetune the networks for different modalities. 
		Please change os.environ['CUDA_VISIBLE_DEVICES'] according to your workstation. 
	5.one/five-shot learning test
		Use ne_shot_test.py and five_shot_test.py to evaluate the performance

Detailed description: 
Files:
	dataset_splits: a file saves the training_list and test_list
	new_dataset: 29 classes of gestures
	one_shot_test_data: register_samples,negative_samples and test_samples
	pretrained_models: initialization parameters(pre-trained from IsoGd dataset)
   
1. network training:
	tensorlayer-layers.py: The original Tensorlayer does not support the convolutional LSTM, 
		so the tensorlayer/layers.py needs to be replaced with tensorlayer-layers.py. 
	c3d_biclstm.py: network architecture
	ConvLSTMCell.py: Convolutional LSTM network cell (ConvLSTM)
	inputs.py: process the data
		input:  image_info = zip(image_path,image_fcnt,image_olen,is_training)
		output: processed gesture (32, 112, 112, 3)
	training_isogr_depth.py: run the file to train network

2. one/five shot learning test
	extract_features.py: extract features using pre-trained model(19 classes of gestures)
		input: the list of register_samples,negative_samples and test_samples
		output: the features of register_samples,negative_samples and test_samples
	one_shot_test.py: test the performance of one-shot learning
		input: the features of register_samples(10*1),negative_samples and test_samples
		output: the performance such as precision,recall,F1-score and confusion matrix
	five_shot_test.py: test the performance of five-shot learning
		input: the features of register_samples(10*5),negative_samples and test_samples
		output: the performance such as precision,recall,F1-score and confusion matrix
	online_test_lxj.py: online one-shot learning
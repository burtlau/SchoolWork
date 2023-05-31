Instructions for CSC420_A2_part2.ipynb

- Run the cell step by step

- Replace the dataset with the dataset you use.

- And reassign the model using the CNN model you want

- You can also change the parameter setting here
	num_epochs = 10
	opt_func = torch.optim.SGD

	max_lr = 0.01
	grad_clip = 0.1
	weight_decay = 1e-4

- note: the ResNet 18 I used is pertained, if you want not pretrained model, simply set the 
	self.network = models.resnet18(pretrained=True) to self.network = models.resnet18(pretrained=False)

- note: the DogBreedClassificationCNN is without drop_out layer, if you want with drop out layer, simply uncomment the line # nn.Dropout(0.5)

- note: the results showed in my ipynb file may perform differently on my report. And I only showed DBI of ResNet18, ResNet34, ResNetXt32, for SSD, just rewrite the dataset to dataset_SSD. 
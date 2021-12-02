==========================================================================================================================================
Training on Public training dataset
---------------------------------------------------

Training is done on the public training dataset of CIFAR-10.

Download the dataset from https://www.cs.toronto.edu/˜kriz/cifar.html

To start training, run the below command.

!python main.py --mode=train --batch_size=128

The following configurations can be changed and added in the above command:
1.  mode            - Denotes whether the program will be run for train, test, or prediction on private dataset. Default is "train".
2.  data_dir        - Path of the data directory. Default is "../Data".
3.  batch_size      - Denotes the batch size during training. Default is 128.
4.  save_dir        - Path to save the models during training. Default is "../model_dir".
5.  result_dir      - Path to save the prediction result of the private dataset. Default is "../results".
6.  weight_decay    - Denotes weight decay to be used in the Optimizer. Default is 5e-4.
7.  save_interval   - Denotes the epoch interval in which the model is saved. Default is 5.
8.  private_model   - Path to the best model that can be used for private dataset predictions. Default is "../saved_models".
9.  checkpoint_name - Name of the best model after all hyperparameter tuning. Default is "model-dpn".
10. checkpoint_list - List of checkpoints that can be used for testing purposes. Default is "50,75,100"

==========================================================================================================================================
Testing on Public test dataset
---------------------------------------------------

Testing is done on the public testing dataset of CIFAR-10.

To start testing, run the below command.

!python main.py --mode=test --checkpoint_list=70,80,90,100

==========================================================================================================================================
Prediction on Private dataset
---------------------------------------------------

Once the model is trained and tested on the public dataset, 

1. Copy the final model in ckpt format from the model_dir folder or the folder name passed in the --save_dir configuration and save it 
into saved_models folder or the folder name passed in the --private-model configuration. 

2. Change the name into "model-dpn" and run the below command.
!python main.py --mode=predict --checkpoint_name=model-dpn

If you don't want to change the model name in the Step 2, then pass the model name in the --checkpoint_name configuration.s
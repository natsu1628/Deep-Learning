# This file is not used. All configurations can be passed as arguments and README file has the details on the
# hyperparameters usage


model_configs = {
	"name": 'CifarDL',
	"save_dir": '../model_dir/',
	"result_dir": '../results/',
	"data_dir": '../Data/',
	"weight_decay": 5e-4,
	"batch_size": 128
}

training_configs = {
	"learning_rate": 0.01,
	"save_interval": 5
}

testing_configs = {
	"checkpoint_list": "50,60,75,80,90,100"
}

prediction_configs = {
	"private_model": '../saved_models/',
	"checkpoint_name": 'model-dpn',
	"result_dir": './'
}

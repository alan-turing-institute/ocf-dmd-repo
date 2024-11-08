from cloudcasting.models import AbstractModel
from dmd_routines import reshape_data2dmd, train_dmd, reshape_Psi2data, eval_dmd

import ocf_dmd_repo

# We define a new class that inherits from AbstractModel
class DmdOneChannel(AbstractModel):
    """DmdOneChannel model class"""

    def __init__(self, history_steps: int, rank: int = 3, time_delay: int = 2) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data.
        # The history_steps parameter should be specified in `validate_config.yml`, along with
        # any other parameters (replace `example_parameter` with as many other parameters as you need to initialize your model, and also add them to `validate_config.yml` under `model: params`)
        super().__init__(history_steps)


        ###### YOUR CODE HERE ######
        # Here you can add any other parameters that you need to initialize your model
        # You might load your trained ML model or set up an optical flow method here.
        # You can also access any code from src/ocf_dmd_repo, e.g.
        x = ocf_dmd_repo.load_model()
        self.rank = rank
        self.time_delay = time_delay

        ############################


    def forward(self, X):
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)

        ###### YOUR CODE HERE ######
        # Grab the most recent frame from the input data
        samples_list = []

        T_train = X.shape[2]
        t_train = np.arange(-T_train, 0)
        for b in range (X.shape[0]): 
            channel_list = []

            for image in range (X.shape[1]): 

                # cropping the horizon
                X0 = image[:, :290, :]
                
                # time delay and dmd snapshots
                X_delayed, t_delayed, data_shape = reshape_data2dmd(X0, 
                                                                    t_train, 
                                                                    time_delay = self.time_delay, 
                                                                    mask = None,
                                                                    isKeepFirstTimes = True)
                
                # train dmd
                Lambda, Psi_, bn = train_dmd(X_delayed, t_delayed, svd_rank = self.rank, 
                                             eig_constraints={
                                                "stable", # choose Re(lambda)<0
                                                "conjugate_pairs", # force complex conjugate pairs
                                                } 
                )

                # reshape Psi to non-time-lagged spacial domain
                Psi = reshape_Psi2data(Psi_, data_shape, mask = None) # reshape to non-time-delayed shape

                t_pred = np.arange(0, 11)
                X_pred = eval_dmd(Lambda, Psi, bn, t_pred, isPositive = True)

                image_pred = np.empty_like(image)
                image_pred[:] = np.nan
                image_pred[:, :290, :] = X_pred

                
                channel_list.append(image_pred)
                
            channel_stack = np.stack(channel_list)
            samples_list.append(channel_stack)
            
        samples_stack = np.stack(samples_list)
        
        return samples_stack
    
    def hyperparameters_dict(self):

        # This function should return a dictionary of hyperparameters for the model
        # This is just for your own reference and will be saved with the model scores to wandb

        ###### YOUR CODE HERE ######
        params_dict =  {
            "rank": self.rank,
            "time_delay": self.time_delay,
        }

        ############################

        return params_dict
      
      
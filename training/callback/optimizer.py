import copy
from experiment.experiment import s2c
from training.event.handler import EventHandler
from training.clutils import zerolike_params_dict, normalize_blocks, copy_params_dict


class Optimizer(EventHandler):
    """
    Optimizer is the main event handler for optimizers. Just pass a PyTorch scheduler together with its arguments in the
    configuration file.
    """
    __name__ = 'optimizer'

    def __init__(self, model, optimizer_class_name, accumulate_gradients=False, **kwargs):
        super().__init__()
        self.optimizer = s2c(optimizer_class_name)(model.parameters(), **kwargs)
        self.accumulate_gradients = accumulate_gradients

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state
        :param state_dict: the optimizer state
        :return:
        """
        self.optimizer.load_state_dict(state_dict)

    def on_fit_start(self, state):
        """
        Load scheduler from state if any
        :param state: the shared State object
        """
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state.optimizer_state)

    def on_training_epoch_start(self, state):
        """
        Zeroes the gradient at the start of each epoch if gradient needs to be accumulated
        :param state: the shared State object
        """
        if self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_start(self, state):
        """
        Zeroes the gradient at the start of each (mini-)batch if gradient does not need to be accumulated
        :param state: the shared State object
        """
        if not self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        """
        Optimized the model at the end of each (mini-)batch if gradient does not need to be accumulated
        :param state: the shared State object
        """
        if not self.accumulate_gradients:
            self.optimizer.step()

    def on_training_epoch_end(self, state):
        """
        Stores the optimizer at the end of each epoch. If gradient needs to be accumulated performs an optimization step
        :param state: the shared State object
        """
        if self.accumulate_gradients:
            self.optimizer.step()

    def on_epoch_end(self, state):
        """
        Stores the optimizer at the end of each epoch
        :param state: the shared State object
        """
        state.update(optimizer_state=copy.deepcopy(self.optimizer.state_dict()))


class CGMMOptimizer(EventHandler):
    def __init__(self, **kwargs):
        super().__init__()

    def on_eval_epoch_start(self, state):
        """
        Use the "compute_intermediate_outputs" field of the state to decide whether to compute statistics or not during
        this evaluation epoch
        :param state: the shared State object
        """
        cgmm = state.model
        cgmm.compute_intermediate_outputs = state.compute_intermediate_outputs

    # Not necessary, but it may help to debug
    def on_eval_epoch_end(self, state):
        """
        Reset the "compute_intermediate_outputs" field to False
        :param state:
        :return:
        """
        cgmm = state.model
        cgmm.compute_intermediate_outputs = False

    def on_training_epoch_end(self, state):
        """
        Calls the M_step to update the parameters
        :param state: the shared State object
        :return:
        """
        state.model.m_step()


class EWCOptimizer(Optimizer):
    """
    Optimizer with no step for Elastic Weight Consolidation
    """
    __name__ = 'optimizer'

    def __init__(self, model, optimizer_class_name, ewc_mode, normalize_importance, accumulate_gradients=False, **kwargs):
        super(EWCOptimizer, self).__init__(model, optimizer_class_name, accumulate_gradients)
        self.optimizer = s2c(optimizer_class_name)(model.parameters(), **kwargs)
        self.ewc_mode = ewc_mode
        self.normalize_importance = normalize_importance
        self.accumulate_gradients = accumulate_gradients
        self.importance = zerolike_params_dict(model)
        self.num_batches = 0

    def on_training_batch_end(self, state):
        assert hasattr(self, 'is_importance_mode'), "You have to define the attribute is_importance_mode in your exp."

        if self.is_importance_mode:
            self.num_batches += 1

            for (k1,p),(k2,imp) in zip(state.model.named_parameters(), self.importance):
                assert(k1==k2)
                imp += p.grad.data.clone().pow(2)
        else:
            super().on_training_batch_end(state)

    def on_training_epoch_end(self, state):
        assert hasattr(self, 'is_importance_mode'), "You have to define the attribute is_importance_mode in your exp."

        # This is useful only when accumulate_gradients is True
        if not self.is_importance_mode:
            super().on_training_epoch_end(state)

    def on_fit_end(self, state):
        assert hasattr(self, 'is_importance_mode'), "You have to define the attribute is_importance_mode in your exp."
        assert hasattr(state.model, 'importances'), "You have to define the attribute importances in the model."
        assert hasattr(state.model, 'saved_parameters'), "You have to define the attribute saved_parameters in the model."

        if self.is_importance_mode:
            for _, imp in self.importance:
                imp /= float(self.num_batches)

            # max-min normalization among every parameter group
            if self.normalize_importance:
                self.importance = normalize_blocks(self.importance)

            if self.ewc_mode == 'separate' or len(state.model.importances) == 0: # separate or first task
                state.model.importances.append(self.importance)
            elif self.ewc_mode == 'sum' and len(state.model.importances) > 0: # sum or > first task
                for (k1,curr_imp),(k2,imp) in zip(self.importance, state.model.importances[0]):
                    assert(k1==k2)
                    imp += curr_imp
            else:
                raise NotImplementedError(f'ewc_mode {self.ewc_mode} not recognized')
        else:
            if self.ewc_mode == 'sum':
                state.model.saved_parameters = [copy_params_dict(state.model)]
            elif self.ewc_mode == 'separate':
                state.model.saved_parameters.append(copy_params_dict(state.model))

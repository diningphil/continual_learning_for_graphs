import time
import operator
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.modules.loss import MSELoss, L1Loss, BCELoss
from training.event.handler import EventHandler


class Loss(nn.Module, EventHandler):
    """
    Loss is the main event handler for loss metrics. Other losses can easily subclass by implementing the forward
    method, though sometimes more complex implementations are required.
    """
    __name__ = "loss"
    op = operator.lt  # less than to determine improvement

    def __init__(self):
        super().__init__()
        self.batch_losses = None
        self.num_targets = None

    def get_main_loss_name(self):
        return self.__name__

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_targets = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_training_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum()/self.num_targets})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_targets = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum()/self.num_targets})
        self.batch_losses = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        loss_output = self.forward(targets, *outputs)

        if isinstance(loss_output, tuple):
            # Allow loss to produce intermediate results that speed up
            # Score computation. Loss callback MUST occur before the score one.
            loss, extra = loss_output
            state.update(batch_loss_extra={self.__name__: extra})
        else:
            loss = loss_output
        state.update(batch_loss={self.__name__: loss})

    def on_backward(self, state):
        """
        Computes the gradient of the computation graph
        :param state: the shared State object
        """
        try:
            state.batch_loss[self.__name__].backward()
        except Exception as e:
            # Here we catch potential multiprocessing related issues
            # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
            print(e)

    def forward(self, targets, *outputs):
        """
        Computes the loss for a batch of output/target valies
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: loss and accuracy values
        """
        raise NotImplementedError('To be implemented by a subclass')


class AdditiveLoss(Loss):
    """
    MultiLoss combines an arbitrary number of Loss objects to perform backprop without having to istantiate a new class.
    The final loss is formally defined as the sum of the individual losses.
    """
    __name__ = "Additive Loss"
    op = operator.lt  # less than to determine improvement

    def _istantiate_loss(self, loss):
        if isinstance(loss, dict):
            args = loss["args"]
            return s2c(loss['class_name'])(*args)
        else:
            return s2c(loss)()

    def __init__(self, **losses):
        super().__init__()
        self.losses = [self._istantiate_loss(loss) for loss in losses.values()]

    def on_training_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_training_batch_end(self, state):
        for k,v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_training_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum()/self.num_targets
                                  for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_eval_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum() / self.num_targets
                          for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_batch_end(self, state):
        for k,v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        loss = {}
        extra = {}
        loss_sum = 0.
        for l in self.losses:
            single_loss = l.forward(targets, *outputs)
            if isinstance(single_loss, tuple):
                # Allow loss to produce intermediate results that speed up
                # Score computation. Loss callback MUST occur before the score one.
                loss_output, loss_extra = single_loss
                extra[single_loss.__name__] = loss_extra
                state.update(batch_loss_extra=extra)
            else:
                loss_output = single_loss
            loss[l.__name__] = loss_output
            loss_sum += loss_output

        loss[self.__name__] = loss_sum
        state.update(batch_loss=loss)


class ClassificationLoss(Loss):
    __name__ = 'Classification Loss'

    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]

        # print(outputs.shape, targets.shape)

        loss = self.loss(outputs, targets)
        return loss


class RegressionLoss(Loss):
    __name__ = 'Regression Loss'


    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs.squeeze(), targets.squeeze())
        return loss


class BinaryClassificationLoss(ClassificationLoss):
    __name__ = 'Binary Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)


class MulticlassClassificationLoss(ClassificationLoss):
    __name__ = 'Multiclass Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


class MeanSquareErrorLoss(RegressionLoss):
    __name__ = 'MSE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = MSELoss(reduction=reduction)


class MeanAverageErrorLoss(RegressionLoss):
    __name__ = 'MAE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = L1Loss(reduction=reduction)


class CGMMLoss(Loss):
    __name__ = 'CGMM Loss'

    def __init__(self):
        super().__init__()
        self.old_likelihood = -float('inf')
        self.new_likelihood = None
        self.training = None

    def on_training_epoch_start(self, state):
        super().on_training_epoch_start(state)
        self.training = True

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)
        self.training = False

    # Simply ignore targets
    def forward(self, targets, *outputs):  # IMPORTANT: This method assumes the batch size is the size of the dataset
        likelihood = outputs[0]

        if self.training:
            self.new_likelihood = likelihood

        return likelihood

    def on_backward(self, state):
        pass

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)

        if (self.new_likelihood - self.old_likelihood) <= 0:
            state.stop_training = True
        self.old_likelihood = self.new_likelihood


class LinkPredictionLoss(Loss):
    __name__ = 'Link Prediction Loss'

    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        loss =  torch.nn.functional.binary_cross_entropy_with_logits(link_logits, loss_target)
        return loss


class EWCLoss(Loss):
    """
    Loss associated to EWC
    cross entropy + lambda * ewc_penalization(importance, current_param, old_param)
    """
    __name__ = 'EWC Loss'

    def __init__(self, ewc_lambda):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """

        outputs, targets = state.batch_outputs, state.batch_targets
        loss_output = self.forward(targets, outputs[0], state.model)

        if isinstance(loss_output, tuple):
            # Allow loss to produce intermediate results that speed up
            # Score computation. Loss callback MUST occur before the score one.
            loss, extra = loss_output
            state.update(batch_loss_extra={self.__name__: extra})
        else:
            loss = loss_output
        state.update(batch_loss={self.__name__: loss})

    def forward(self, targets, output, model):
        clf_loss = self.loss(output, targets)

        importances = model.importances
        n_importances = len(model.importances)
        penalty = 0.

        if n_importances > 0:
            for task in range(n_importances):
                for (k1, param), (k2, saved_param), (k3, imp) in zip(model.named_parameters(), model.saved_parameters[task], importances[task]):
                    assert(k1==k2==k3)
                    penalty += (imp * (param - saved_param).pow(2)).sum()
        else:
            return clf_loss

        return clf_loss + self.ewc_lambda * penalty


class ClassificationAdjacencyRegularizationLoss(Loss):
    __name__ = 'Classification Adjacency Regularization Loss'

    def __init__(self, adj_lambda):
        super().__init__()
        self.adj_lambda = adj_lambda
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, targets, *outputs):
        node_embs = outputs[1]
        edge_index = outputs[2]

        src_embs = node_embs[edge_index[0]]
        dst_embs = node_embs[edge_index[1]]

        classification_loss = self.loss(outputs[0], targets)
        regularization_loss = torch.norm(src_embs - dst_embs, p=2, dim=1).mean(dim=0)
        return classification_loss + self.adj_lambda*regularization_loss


class LwFLoss(Loss):
    __name__ = 'LwF Loss'

    def __init__(self, alpha=0.5, distillation_temperature=2, mode='sum'):
        """ Learning without Forgetting.

        paper: https://arxiv.org/abs/1606.09282
        original implementation (Matlab):
            https://github.com/lizhitwo/LearningWithoutForgetting
        reference implementation (pytorch):
            https://github.com/arunmallya/packnet/blob/master/src/lwf.py

        :param alpha: distillation loss coefficient.
        :param distillation_temperature: distillation loss temperature.
        :param warmup_epochs: number of warmup epochs training only
            the new parameters.
        """
        super().__init__()
        assert mode in {'sum', 'last'}
        self.distillation_temperature = distillation_temperature
        self.alpha = alpha
        self.mode = mode
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        self.prev_models = state.prev_models
        self.curr_batch = state.batch_input
        super().on_compute_metrics(state)

    def forward(self, targets, *outputs):
        y_mb = targets
        logits = outputs[0]
        loss = self.loss(logits, y_mb)

        task_id = len(self.prev_models)
        if task_id == 0:
            return loss

        if self.mode == 'sum':
            for teacher in self.prev_models:
                y_teacher = teacher(self.curr_batch)[0].detach()
                dist_loss = self.distillation_loss(logits, y_teacher)
                loss += self.alpha * dist_loss
        elif self.mode == 'last':
            curr_alpha = self.alpha * task_id * (task_id / (task_id + 1))  # 0, 1*1/2, 2*2/3, 3*3/4, ...
            teacher = self.prev_models[-1]
            y_teacher = teacher(self.curr_batch)[0].detach()
            dist_loss = self.distillation_loss(logits, y_teacher)
            loss += curr_alpha * dist_loss
        else:
            assert False
        return loss

    def distillation_loss(self, y_pred, y_teacher):
        """ Distillation loss. """
        # kl_div is normalized by element instead of observation
        temperature = self.distillation_temperature
        scale = y_teacher.shape[-1]
        log_p = F.log_softmax(y_pred / temperature, dim=1)
        q = F.softmax(y_teacher / temperature, dim=1)
        res = scale * F.kl_div(log_p, q, reduction='mean')
        return res

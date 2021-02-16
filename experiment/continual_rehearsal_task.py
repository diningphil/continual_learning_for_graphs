import os
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import DataLoader

from experiment.experiment import Experiment
from data.dataset import ConcatFromListDataset


class ContinualRehearsalTask(Experiment):
    """
    We assume a classification task
    """
    def __init__(self, model_configuration, exp_path):
        super(ContinualRehearsalTask, self).__init__(model_configuration, exp_path)
        self.write_training_header = True
        self.write_intermediate_header = True


    def write_training_metrics(self, t, train_loss, train_score, val_loss, val_score):

        with open(os.path.join(self.orig_exp_path, 'training_results.csv'), 'a') as csvfile:

            if self.write_training_header:
                tr_names = ",".join([str(el) for el in train_score.keys()])
                val_names = ",".join([str(el) for el in val_score.keys()])
                csvfile.write(f"task_id,train_loss,{tr_names},val_loss,{val_names}\n")
                self.write_training_header = False

            tv = ",".join([str(float(el)) for el in train_score.values()])
            vv = ",".join([str(float(el)) for el in val_score.values()])

            csvfile.write(f"{t},{train_loss['main_loss'].item()},{tv},{val_loss['main_loss'].item()},{vv}\n")


    def write_intermediate_metrics(self, prev_t_id, t, loss, score):

        with open(os.path.join(self.orig_exp_path, 'intermediate_results.csv'), 'a') as csvfile:

            if self.write_intermediate_header:
                names = ",".join([str(el) for el in score.keys()])
                csvfile.write(f"prev_task_id,task_id,loss,{names}\n")
                self.write_intermediate_header = False

            v = ",".join([str(float(el)) for el in score.values()])

            csvfile.write(f"{prev_t_id},{t},{loss['main_loss'].item()},{v}\n")


    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation scores
        :return: (training score, validation score)
        """
        self.orig_exp_path = self.exp_path

        # In case we are resuming execution from a checkpoint
        if os.path.exists(os.path.join(self.orig_exp_path, 'intermediate_results.csv')):
            os.remove(os.path.join(self.orig_exp_path, 'intermediate_results.csv'))
        if os.path.exists(os.path.join(self.orig_exp_path, 'training_results.csv')):
            os.remove(os.path.join(self.orig_exp_path, 'training_results.csv'))

        n_tasks = self.model_config.n_tasks
        n_rehearsal_patterns_per_task = self.model_config.n_rehearsal_patterns_per_task
        assert n_tasks is not None, "You must provide a n_tasks value in the configuration file."

        batch_size = self.model_config.supervised_config['batch_size']
        shuffle = self.model_config.supervised_config['shuffle'] \
            if 'shuffle' in self.model_config.supervised_config else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features, dim_edge_features, dim_target)

        rehearsal_memory = []
        avg_val_score = defaultdict(float)
        for t in range(n_tasks):

            ### TRAIN CURRENT TASK ###

            self.exp_path = Path(self.orig_exp_path, f'task_{t}')

            # Read sequentially ow it breaks (Pytorch fault??)
            num_workers, pin_memory = dataset_getter.num_workers, dataset_getter.pin_memory
            dataset_getter.num_workers, dataset_getter.pin_memory = 0, False
            orig_train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle, task_id=t)
            dataset_getter.num_workers, dataset_getter.pin_memory = num_workers, pin_memory

            val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle, task_id=t)

            data_list = [el for batch in orig_train_loader for el in batch.to_data_list()] + rehearsal_memory
            concat_dataset = ConcatFromListDataset(data_list)
            train_loader = DataLoader(concat_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=dataset_getter.num_workers, pin_memory=dataset_getter.pin_memory)

            # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
            supervised_training_wrapper = self.create_supervised_wrapper(model)

            train_loss, train_score, _, \
            val_loss, val_score, _, \
            _, _, _ = supervised_training_wrapper.train(
                                                        train_loader=train_loader,
                                                        validation_loader=val_loader,
                                                        test_loader=None,
                                                        max_epochs=self.model_config.supervised_config['epochs'],
                                                        logger=logger)

            self.write_training_metrics(t, train_loss, train_score, val_loss, val_score)

            #### UPDATE REHEARSAL PATTERNS ###

            # Default is 0
            class_counter = defaultdict(int)
            # Batch size is 1, shuffle depends on the config file
            for batch in orig_train_loader:
                for d in batch.to_data_list():
                    c = d.y.item()
                    if class_counter[c] < n_rehearsal_patterns_per_task:
                        class_counter[c] += 1
                        rehearsal_memory.append(d)

            #### EVALUATE PREVIOUS TASKS ###

            for prev_t_id in range(t+1):
                val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle, task_id=prev_t_id)
                val_loss, val_score, _ = supervised_training_wrapper.infer(val_loader, set=supervised_training_wrapper.VALIDATION)

                self.write_intermediate_metrics(prev_t_id, t, val_loss, val_score)
                if t == n_tasks - 1:
                    for k in val_score.keys():
                        avg_val_score[k] += val_score[k]


        # avg_train_score not used
        avg_train_score = dict.fromkeys(val_score, -1)
        for k in avg_val_score.keys():
            avg_val_score[k] /= float(n_tasks)

        return avg_train_score, avg_val_score

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test score. DO NOT USE THE TEST TO TRAIN OR FOR EARLY STOPPING REASONS!
        :return: (training score, test score)
        """
        self.orig_exp_path = self.exp_path

        # In case we are resuming execution from a checkpoint
        if os.path.exists(os.path.join(self.orig_exp_path, 'intermediate_results.csv')):
            os.remove(os.path.join(self.orig_exp_path, 'intermediate_results.csv'))
        if os.path.exists(os.path.join(self.orig_exp_path, 'training_results.csv')):
            os.remove(os.path.join(self.orig_exp_path, 'training_results.csv'))

        n_tasks = self.model_config.n_tasks
        n_rehearsal_patterns_per_task = self.model_config.n_rehearsal_patterns_per_task
        assert n_tasks is not None, "You must provide a n_tasks value in the configuration file."

        batch_size = self.model_config.supervised_config['batch_size']
        shuffle = self.model_config.supervised_config['shuffle'] \
            if 'shuffle' in self.model_config.supervised_config else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features, dim_edge_features, dim_target)

        rehearsal_memory = []
        avg_test_score = defaultdict(float)
        for t in range(n_tasks):

            self.exp_path = Path(self.orig_exp_path, f'task_{t}')

            ### TRAIN CURRENT TASK ###

            num_workers, pin_memory = dataset_getter.num_workers, dataset_getter.pin_memory
            dataset_getter.num_workers, dataset_getter.pin_memory = 0, False
            orig_train_loader = dataset_getter.get_outer_train(batch_size=1, shuffle=shuffle, task_id=t)
            dataset_getter.num_workers, dataset_getter.pin_memory = num_workers, pin_memory

            val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle, task_id=t)
            test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle, task_id=t)

            data_list = [el for batch in orig_train_loader for el in batch.to_data_list()] + rehearsal_memory
            concat_dataset = ConcatFromListDataset(data_list)
            train_loader = DataLoader(concat_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=dataset_getter.num_workers, pin_memory=dataset_getter.pin_memory)

            # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
            supervised_training_wrapper = self.create_supervised_wrapper(model)

            train_loss, train_score, _, \
            val_loss, val_score, _, \
            test_loss, test_score, _ = supervised_training_wrapper.train(
                                                                        train_loader=train_loader,
                                                                        validation_loader=val_loader,
                                                                        test_loader=test_loader,
                                                                        max_epochs=self.model_config.supervised_config['epochs'],
                                                                        logger=logger)



            self.write_training_metrics(t, train_loss, train_score, test_loss, test_score)

            #### UPDATE REHEARSAL PATTERNS ###

            # Default is 0
            class_counter = defaultdict(int)
            # Batch size is 1, shuffle depends on the config file
            for batch in orig_train_loader:
                for d in batch.to_data_list():
                    c = d.y.item()
                    if class_counter[c] < n_rehearsal_patterns_per_task:
                        class_counter[c] += 1
                        rehearsal_memory.append(d)

            #### EVALUATE PREVIOUS TASKS ###

            for prev_t_id in range(t+1):
                test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle, task_id=prev_t_id)
                test_loss, test_score, _ = supervised_training_wrapper.infer(test_loader, set=supervised_training_wrapper.TEST)

                self.write_intermediate_metrics(prev_t_id, t, test_loss, test_score)
                if t == n_tasks - 1:
                    for k in test_score.keys():
                        avg_test_score[k] += test_score[k]

        # avg_train_score not used
        avg_train_score = dict.fromkeys(test_score, -1)
        for k in avg_test_score.keys():
            avg_test_score[k] /= float(n_tasks)

        return avg_train_score, avg_test_score

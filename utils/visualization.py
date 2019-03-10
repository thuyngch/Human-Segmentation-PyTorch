#------------------------------------------------------------------------------
#    Libraries
#------------------------------------------------------------------------------
import importlib
import warnings
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#    Tensorboard Writer
#------------------------------------------------------------------------------
class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn(message)
        self.step = 0
        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step):
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}'.format(tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


#------------------------------------------------------------------------------
#   Function to plot Tensorboard
#------------------------------------------------------------------------------
def plot_tensorboard(train_file, valid_file, scalar_names, set_grid=False):
    # Read Tensorboard files
    train_event_acc = EventAccumulator(train_file)
    valid_event_acc = EventAccumulator(valid_file)
    train_event_acc.Reload()
    valid_event_acc.Reload()

    # Get scalar values
    train_scalars, valid_scalars = {}, {}
    for scalar_name in scalar_names:
        train_scalars[scalar_name] = train_event_acc.Scalars(scalar_name)
        valid_scalars[scalar_name] = valid_event_acc.Scalars(scalar_name)

    # Convert to list
    n_epochs = len(train_scalars["loss"])
    epochs = [train_scalars["loss"][i][1] for i in range(n_epochs)]

    train_lists, valid_lists = {}, {}
    for scalar_name in scalar_names:
        train_lists[scalar_name] = [train_scalars[scalar_name][i][2] for i in range(n_epochs)]
        valid_lists[scalar_name] = [valid_scalars[scalar_name][i][2] for i in range(n_epochs)]

    # Plot
    for scalar_name in scalar_names:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if set_grid:
            ax.set_xticks(epochs)

        ax.plot(epochs, train_lists[scalar_name], label='train')
        ax.plot(epochs, valid_lists[scalar_name], label='valid')

        plt.xlabel("epochs")
        plt.ylabel(scalar_name)
        plt.legend(frameon=True)
        plt.grid(True)
        plt.show()


#------------------------------------------------------------------------------
#   Test bench
#------------------------------------------------------------------------------
if __name__ == '__main__':
    train_file = "checkpoints/runs/Mnist_LeNet/1125_110943/train/events.out.tfevents.1543118983.antiaegis"
    valid_file = "checkpoints/runs/Mnist_LeNet/1125_110943/valid/events.out.tfevents.1543118983.antiaegis"
    plot_tensorboard(train_file, valid_file, ["loss", "my_metric", "my_metric2"])
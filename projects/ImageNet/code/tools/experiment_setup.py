
""" Setup an experiment directory (as a self-contained module) for each training run """


import os


def create_train_test_subdir(main_dir, include_val=False):
    """ Given a directory, create train and test subdirectories within it """
    train_dir = os.path.join(main_dir, "train", "")
    os.mkdir(train_dir)
    test_dir = os.path.join(main_dir, "test", "")
    os.mkdir(test_dir)

    if include_val:
        val_dir = os.path.join(main_dir, "val", "")
        os.mkdir(val_dir)


def setup_exp_directory(run_id=None, write_dir=None):
    """ Create experiment directory and all key subdirectories """
    if write_dir is None:
        # Get absolute path of the working directory that Python is currently running in
        write_dir = os.getcwd()
        write_dir = write_dir.replace("code", "experiments")

    # Create experiment directory
    exp_dir = os.path.join(write_dir, run_id, "")
    os.mkdir(exp_dir)
    print("Experiment directory: \n" + exp_dir)

    # Create subdirectories
    # Logs - with train and test subdirs
    logs_dir = os.path.join(exp_dir, "logs", "")
    os.mkdir(logs_dir)
    create_train_test_subdir(logs_dir)

    # Visualisations - with train and test subdirs
    vis_dir = os.path.join(exp_dir, "vis", "")
    os.mkdir(vis_dir)
    create_train_test_subdir(vis_dir, include_val=True)

    # Models
    models_dir = os.path.join(exp_dir, "models", "")
    os.mkdir(models_dir)

    # Code
    code_dir = os.path.join(exp_dir, "code", "")
    os.mkdir(code_dir)
    os.system("cp -r * " + str(code_dir))     # copy current code into the experiment directory

    return exp_dir, logs_dir, vis_dir, models_dir, code_dir


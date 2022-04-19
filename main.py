# suppress warning while using CPU
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import run_lib
import tensorflow as tf

from absl import flags, app
from ml_collections import config_flags

FLAGS = flags.FLAGS

# read configuration file
config_flags.DEFINE_config_file("config", None, "Training configuration", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])



def main(argv):
    # print(FLAGS.config)

    if FLAGS.mode == "train":
        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        print("File created at :", FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        print("\tTraining model")
        # run training passing configuration file parameters and working directory
        run_lib.train(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)



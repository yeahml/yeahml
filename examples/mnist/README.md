[//]: # (Image References)
[example_run]: ./misc/example_run.png

# MNIST

The current example expects the tf records to already be included on the local environment (these can be created by using the included [`.py` script](./make_records/convert_to_records.py)). After the tf records have been created. The current directory and config files should work as expected when calling `python yamlflow.py` with the config paths set correctly.

## Sample Instructions

#### Get the Data, Create TF records
1. open terminal and navigate to the `/mnist/make_records/` directory
    - e.g. `cd /mnist/make_records/` (if on mac/linux system, if using windows, `dir /mnist/make_records/`)
2. Run the included script to make the tf records (they will be placed in the `/mnist/make_records/` directory by default)
    - Note: this will download and extract mnist files and will require an internet connection. You will also want to ensure your python environment has tensorflow installed.
    - e.g. `python convert_to_records.py`
3. Ensure the tfrecords are present in the `mnist/data/` directory (the .gz files can be deleted)
    - e.g. `cd ../data/`

#### Ensure the correct example is selected
1. open the main example file (`run_example.py`)
2. ensure the line `example = "./examples/mnist/model_config.yaml"  # softmax example` is uncommented and the other `example = ...` lines are commented

#### Run example (and experiment with configuration files)
1. `python run_example.py`

![Example of the project running in the terminal][example_run]

## Questions
Feel free to reach out to me on twitter and I will try to help ([@Jack_Burdick](https://twitter.com/Jack_Burdick))



## NOTE

This directory is used as the development data for `softmax`

Run times for this example should be relatively fast. On an CPU (2015 MBP i7), each iteration is roughly 111 seconds.
# [NuralNet](https://cseegit.essex.ac.uk/mh16185/ce601/tree/master/neuralnet)

## [Config](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Config.py)
This file contains some constant values used throughout the app. It contains the following values:
- `BATCH_SIZE`
- `EPOCHS`
- `LATENT_DIM`
- `NUM_SAMPLES` 
- `MAX_LINE_SIZE`

## [DataResult](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataResult.py)
This file contains a class called DataResult, it is the class that represents the result from the [DataService](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py) and contains the following properties:
- `input_data`
- `output_data`

## [DataService](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py)

### [__pre_process_input](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L11)
This method goes through each line in the specified input file and filters out any lines that surpass the `MAX_LINE_SIZE` specified in the config. This is used to filter out any extremely large lines as they cause memory issues when attempting to build up the NumPy array.

### [__pre_process_output](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L18)
This method goes through each line in the specified output file and filters out any lines that surpass the `MAX_LINE_SIZE` specified in the config. This is used to filter out any extremely large lines as they cause memory issues when attempting to build up the NumPy array.

### [__process_input](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L25)
This method is used to loop through each line in the input file and checks that it hasn’t been blacklisted by the [`__pre_process_input`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L11) method. If the line has not been blacklisted it then adds the line to the `lines` property defined on the `input_data` that has been defined in the class. It then loops through each character if it has not already been seen it adds it to the `chars` property on the `input_data` that has been defined in the class.

### [__process_output](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L52)
This method is used to loop through each line in the output file and checks that it hasn’t been blacklisted by the [`__pre_process_output`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L18) method. If the line has not been blacklisted it then adds the line to the `lines` property defined on the `output_data` that has been defined in the class. It then loops through each character if it has not already been seen it adds it to the `chars` property on the `output_data` that has been defined in the class.

### [process](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L79)
This method is used to process the data set and calls to all the other internal methods. First it calls the [`__pre_process_input`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L11) method and then the [`__pre_process_output`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L18) method. Once the pre-processing has occurred, it then calls into the [`__process_input`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L25) method and the [`__process_output`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L52). Once all the methods have been called, it then creates a new [`DataResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataResult.py) and passes in the `input_data` and `output_data` defined from the previous methods.

## [DataTokenizer](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py)

### [__process_indexes](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L14)
This method attempts to index all the characters collected from the input and output results. First it loops through all the characters in the input data and adds them to the `input_index` dictionary, it then repeats this process for the output data and adds entries to the `output_index` dictionary.

### [__init_data](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L28)
This method initialises the NumPy arrays that will be used later in execution. it initialises the following variables as 3d arrays:

- `encoder_input`
- `decoder_input`
- `decoder_output`

### [__load_data](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L55)
This method loops through the input and output lines simultaneously and attempts to populate the `encoder_input`, `decoder_input` and the `decoder_output` using the indexes populated by the previous method [`__process_indexes`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L14).

### [tokenize](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L72)
This method is used to tokenize all the data passed into the service. It makes calls to the method previously defined to produce a [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py). First it makes a call to [`__process_indexes`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L14) to process all the indexes. Once the indexes have been processed it then makes a call to [`__load_data`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L55) to load in all the necessary data. Once the data has been loaded it will then build up a [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py), set the properties and return it.


## [Decoder](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Decoder.py)
This class it used to represent the decoder in the chatbot, it contains the following properties:
- `inputs`
- `lstm`
- `dense`
- `outputs`

## [Encoder](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Encoder.py)
This class it used to represent the encoder in the chatbot, it contains the following properties:
- `inputs`
- `lstm`
- `encoder_outputs`
- `states`

## [InputData](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/InputData.py)
This class is used to hold the information around the input data being passed into the neural network. It contains the following properties:
- `lines`
- `chars`
- `num_tokens`
- `max_len`

## [OutputData](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/OutputData.py)
This class is used to hold the information around the output data being passed into the neural network. It contains the following properties:
- `lines`
- `chars`
- `num_tokens`
- `max_len`

## [TokenResult](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py)
This class is used to hold the information produced by the [`DataTokenizer`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py). It contains the following properties:
- `input_index`
- `output_index`
- `encoder_input`
- `decoder_input`
- `decoder_output`

## [Program](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/program.py)
This file is the main file that will be executed when attempting to train the chatbot. there are a few steps within the program and are as follows:

1. Build up a [`DataResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataResult.py) using the [`process`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py#L79) method as defined in the [`DataService`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataService.py).
2. Build up a [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py) using the [`tokenize`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py#L72) method as defined on the [`DataTokenizer`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataTokenizer.py)
3. Create a new [`Encoder`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Encoder.py) by passing in the `input_data` from the [`DataResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataResult.py) created in step 1
4. Create a new [`Decoder`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Decoder.py) by passing in the `output_data` from the [`DataResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/DataResult.py) created in step 1
5. Create a new `Keras` model by passing in the following parameters:
    - The `inputs` from the [`Encoder`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Encoder.py) created in step 3
    - The `inputs` from the [`Decoder`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Decoder.py) created in step 4
    - The `outputs` from the [`Decoder`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/Decoder.py) created in step 4
6. Compile the model
7. Train the model by calling the `fit` method and passing in the following parameters:
    - The `encoder_input` from the [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py) created in step 2
    - The `decoder_input` from the [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py) created in step 2
    - The `decoder_output` from the [`TokenResult`](https://cseegit.essex.ac.uk/mh16185/ce601/blob/master/neuralnet/TokenResult.py) created in step 2

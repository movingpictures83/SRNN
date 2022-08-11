from TestFloodCas_modules import *
import PyPluMA

class SRNNPlugin:
   def input(self, filename):
       self.parameters = dict()
       infile = open(filename, 'r')
       for line in infile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]

   def run(self):
    scaler = MinMaxScaler(feature_range=(0,1))
    # Get data and create dataset
    data = read_and_slice_dataset(PyPluMA.prefix()+"/"+self.parameters["inputfile"], 0, int(self.parameters["slice"]))
    # Set params for the training
    n_hours, n_features, K = set_lag_hours(int(self.parameters["hours"]), int(self.parameters["features"]), int(self.parameters["K"]))
    # Create stages and non-stages, and supervised
    mystages = []
    stagefile = open(PyPluMA.prefix()+"/"+self.parameters["stagefile"], 'r')
    for line in stagefile:
        mystages.append(line.strip())
    stages = create_set_for_staging(data, mystages)
    #stages = create_set_for_staging(data, ['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26', 'HWS_S25A', 'HWS_S25B', 'HWS_S26'])
    mynonstages = []
    nonstagefile = open(PyPluMA.prefix()+"/"+self.parameters["nonstagefile"], 'r')
    for line in nonstagefile:
        mynonstages.append(line.strip())
    #non_stages = create_set_for_staging(data, ['FLOW_S25A', 'GATE_S25A', 'FLOW_S25B', 'GATE_S25B', 'FLOW_S26', 'GATE_S26', 'PUMP_S26', 'mean'])
    non_stages = create_set_for_staging(data, mynonstages)
    stages_supervised = stage_series_to_supervised(stages, n_hours, K, 1)
    non_stages_supervised = series_to_supervised(non_stages, n_hours, 1)
    stages_df = [stages_supervised, non_stages, non_stages_supervised]
    # Concatenate stages dataframes
    all_data = concat_preprocessed(stages_df)
    # split dataset into train and test sets
    train, test = split_into_train_and_test(all_data)
    # Split train and test sets into X and y
    train_X, train_y, test_X, test_y = split_into_input_and_output(n_hours, n_features, train, test)
    sets = [train_X, train_y, test_X, test_y]
    # Normalize sets with MinMaxScaler
    normalized_sets, scaler = normalize_features(sets)
    # Reshape train_X and test_X
    train_X, test_X = reshape_X_sets(train_X, test_X, n_hours, n_features)

    # GRU PIPELINE
    model = create_RNN_model('Sequential', int(self.parameters["activation"]), int(self.parameters["regression"]), train_X)
    train_RNN(model, float(self.parameters["lr"]), int(self.parameters["epochs"]), train_X, train_y, test_X, test_y)

    
    # Predict stage
    self.inv_y, self.inv_yhat = predict(test_X, test_y, model, scaler)

   def output(self, filename):
    #print(self.inv_y)
    pd.DataFrame(self.inv_y).to_csv(filename+".y.csv")
    pd.DataFrame(self.inv_yhat).to_csv(filename+".yhat.csv")
    #print(self.inv_yhat)
    print("Variables inv_y and inv_yhat contain the results of the prediction. \n")
    print("Program ended successfuly. \n")


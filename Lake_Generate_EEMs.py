##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from functions import preprocess, preprocess_test, mae, r2, peak_pick, tucker

## Import training and testing datasets
data = pd.read_csv(r'Lake_Train_cut.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'Lake_Test_cut.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)

## Normalize train and test dataset
def preprocess(x):
    y = np.copy(x)
    max_v = np.max(x, axis=0)
    min_v = np.min(x, axis=0)
    for i in range(np.size(max_v)):
        if (max_v[i] - min_v[i] > 0):
            y[:, i] = (x[:, i] - min_v[i]) / (max_v[i] - min_v[i])
        else:
            continue
    return y, max_v, min_v  # max and min vectors are needed to renormalize

def preprocess_test(x, max_v, min_v):
    y = np.copy(x)
    for i in range(np.size(max_v)):
        if (max_v[i] - min_v[i] > 0):
            y[:, i] = (x[:, i] - min_v[i]) / (max_v[i] - min_v[i])
        else:
            continue
    return y

input_data_train, max_v, min_v = preprocess(input_data_train)
input_data_test = preprocess_test(input_data_test, max_v, min_v)

# Indexes of the key intensities selected
indexes = [892, 1679, 1819]


## Select few key intensities from the full EEM. The selected intensities correspond to the peak intensities of the contaminants and natural organic matter.
test_decoder = np.zeros((test_quantity, decoder_shape))
train_decoder = np.zeros((train_quantity, decoder_shape))

for i in range(decoder_shape):
    column_test = input_data_test[:, indexes[i]]
    column_train = input_data_train[:, indexes[i]]
    column_test = column_test.T
    column_train = column_train.T
    test_decoder[:, i] = column_test
    train_decoder[:, i] = column_train

# set the input tensor to generate the full EEM
inputs_decoder = layers.Input(shape=(np.shape(train_decoder)[1],))
def decoder(input):
    layer_1 = layers.Dense(units=20)(input)
    layer_1 = layers.BatchNormalization()(layer_1)
    layer_1 = layers.Activation('elu')(layer_1)
    layer_2 = layers.Dense(units=100)(layer_1)
    layer_2 = layers.BatchNormalization()(layer_2)
    layer_2 = layers.Activation('elu')(layer_2)
    layer_3 = layers.Dense(units=500)(layer_2)
    layer_3 = layers.BatchNormalization()(layer_3)
    layer_3 = layers.Activation('elu')(layer_3)
    layer_4 = layers.Dense(units=1000)(layer_3)
    layer_4 = layers.BatchNormalization()(layer_4)
    layer_4 = layers.Activation('elu')(layer_4)
    output = layers.Dense(units=4853, activation='elu')(layer_4)
    return output

# get the model output 
deep_decoder_output = decoder(inputs_decoder)

deep_decoder_model = Model(inputs=inputs_decoder, outputs=deep_decoder_output)
deep_decoder_model.compile(optimizer='adam', loss='mean_squared_error')
deep_decoder_model.summary()


## Train the model
deep_decoder_model.fit(
    x=train_decoder,
    y=input_data_train,
    epochs=300,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

# Generate the EEMs
decoder_predicted_test = deep_decoder_model.predict(test_decoder)
decoder_predicted_train = deep_decoder_model.predict(train_decoder)

# Just an example plot to see how well it worked.
plt.title('7 inputs')
plt.plot(input_data_test[10,300:1000], '--', color='orange')
plt.plot(decoder_predicted_test[10, 300:1000], '-')
plt.show()

## Metrics to evaluate the reuslts. Mean absolute error (MAE), R2, Tucker coefficient.
def r2(real, predicted):
    mean = np.mean(real, axis=0)
    first_errors_autoencoder = np.sum((real-predicted)**2, axis=0)
    second_errors_autoencoder = np.sum((real-mean)**2, axis=0)
    r2_autoencoder = 1-first_errors_autoencoder/second_errors_autoencoder
    r2_autoencoder[r2_autoencoder == -inf] = nan
    r2 = np.nanmean(r2_autoencoder)
    return r2

def tucker(real, predicted):
    tucker_numerador_autoencoder = np.sum(np.multiply(real, predicted), axis=0)
    tucker_demoninator_autoencoder = np.sqrt(np.sum((real)**2, axis=0) * np.sum((predicted)**2, axis=0))
    tucker_autoencoder = tucker_numerador_autoencoder/tucker_demoninator_autoencoder
    tucker = np.nanmean(tucker_autoencoder)
    return tucker

def mae(real, predicted):
    mae = np.sum(abs(real-predicted), axis=0)
    mae= mae/len(real)
    mae = np.nanmean(mae)
    return mae

mae_decoder = float("{:.3f}".format(mae(input_data_test, decoder_predicted_test)))
r2_decoder = float("{:.3f}".format(r2(input_data_test, decoder_predicted_test)))
tucker_decoder = float("{:.3f}".format(tucker(input_data_test, decoder_predicted_test)))

## Import concentrations. The generated EMMs will be used to predict contaminants concentrations.
train_concentration = pd.read_csv(r'Lake_Train_Concentrations.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'Lake_Test_Concentrations.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

## Separate phenol and naphthenic acids concentrations
train_Naph = train_concentration[:,0]
train_Phenol = train_concentration[:,1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:,0]
test_Phenol = test_concentration[:,1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]

## ANN to predict contaminants concentrations from the generated EMMs.
def naph(input):
    layer_1 = layers.Dense(units=1000, activation='elu')(input)
    output = layers.Dense(units=1, activation='elu')(layer_1)
    return output

inputs = layers.Input(shape=(np.shape(decoder_predicted_train)[1],))
naph_output = naph(inputs)
naph_model = Model(inputs=inputs, outputs=naph_output)
naph_model.compile(optimizer='adam', loss='mean_squared_error')

## Train the model
deep_auto_history = naph_model.fit(
    x=decoder_predicted_train,
    y=train_Naph,
    epochs=500,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

## Make naphthenic acids (NAs) predictions
naph_predicted = naph_model.predict(decoder_predicted_test)

## Evaluation metrics for contaminant's quantification
mae_naph = mae(test_Naph, naph_predicted)
r2_naph = r2(test_Naph, naph_predicted)
MSE_naph = np.square(np.subtract(test_Naph, naph_predicted)).mean()
RMSE_naph = float("{:.3f}".format(np.sqrt(MSE_naph)))

real_datafrane = pd.DataFrame(test_concentration, columns=labels)
predicted_dataframe = pd.DataFrame(concentration_predicted, columns=labels)

## Plot predicted concentrations vs real concentrations
plt.figure(figsize=(10,10))
plt.title('1 dense layer', fontsize=20)
plt.scatter(test_Naph, naph_predicted, c='crimson', label="Naphthenic")

x = np.linspace(0, 100)
plt.plot(x, x, c='black', linestyle='dotted')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Measured NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=20)
plt.ylabel('Predicted NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=20)
plt.axis('square')
plt.grid(linestyle='dotted', linewidth=1)
plt.legend(fontsize=20)
plt.ylim(0, 100)
plt.xlim(0, 100)
plt.show()

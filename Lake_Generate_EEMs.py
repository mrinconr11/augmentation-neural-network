##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from functions import preprocess, preprocess_test, mae, r2, peak_pick, tucker
import time
decoder_shape = 7
train_quantity = 144
test_quantity = 36

data = pd.read_csv(r'Lake_Train_cut.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'Lake_Test_cut.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)
##
input_data_train, max_v, min_v = preprocess(input_data_train)
input_data_test = preprocess_test(input_data_test, max_v, min_v)

test_decoder = np.zeros((test_quantity, decoder_shape))
train_decoder = np.zeros((train_quantity, decoder_shape))

indexes = [892, 1679, 1819, 3737, 2340, 4141, 4179]

for i in range(decoder_shape):
    column_test = input_data_test[:, indexes[i]]
    column_train = input_data_train[:, indexes[i]]
    column_test = column_test.T
    column_train = column_train.T
    test_decoder[:, i] = column_test
    train_decoder[:, i] = column_train

# set the input tensor
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
# get the decoder output
deep_decoder_output = decoder(inputs_decoder)

deep_decoder_model = Model(inputs=inputs_decoder, outputs=deep_decoder_output)
deep_decoder_model.compile(optimizer='adam', loss='mean_squared_error')
deep_decoder_model.summary()

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
##
# Just an example plot to see how well it worked.
plt.title('7 inputs')
plt.plot(input_data_test[10,300:1000], '--', color='orange')
plt.plot(decoder_predicted_test[10, 300:1000], '-')
plt.show()
##
# Evaluate the results
mae_decoder = float("{:.3f}".format(mae(input_data_test, decoder_predicted_test)))
r2_decoder = float("{:.3f}".format(r2(input_data_test, decoder_predicted_test)))
tucker_decoder = float("{:.3f}".format(tucker(input_data_test, decoder_predicted_test)))
squared_error = np.mean(np.sqrt((input_data_test - decoder_predicted_test)**2))
##
# Save generated EEMS
test_eem = pd.DataFrame(decoder_predicted_test)
train_eem = pd.DataFrame(decoder_predicted_train)
test_eem.to_csv('Lake_Test_Predicted_7.csv', index=False)
train_eem.to_csv('Lake_Train_Predicted_7.csv', index=False)
y = 0







##
train_concentration = pd.read_csv(r'../CNN/Lake_Train_Concentrations.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'../CNN/Lake_Test_Concentrations.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

train_Naph = train_concentration[:,0]
train_Phenol = train_concentration[:,1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:,0]
test_Phenol = test_concentration[:,1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]
##
def naph(input):
    layer_1 = layers.Dense(units=1000, activation='elu')(input)
    layer_2 = layers.Dense(units=500, activation='elu')(layer_1)
    layer_21 = layers.Dense(units=100, activation='elu')(layer_2)
    layer_3 = layers.Dense(units=50, activation='elu')(layer_21)
    layer_4 = layers.Dense(units=10, activation='elu')(layer_3)
    output = layers.Dense(units=1, activation='elu')(layer_4)
    return output
inputs = layers.Input(shape=(np.shape(decoder_predicted_train)[1],))
naph_output = naph(inputs)
naph_model = Model(inputs=inputs, outputs=naph_output)
naph_model.compile(optimizer='adam', loss='mean_squared_error')
deep_auto_history = naph_model.fit(
    x=decoder_predicted_train,
    y=train_Naph,
    epochs=300,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

naph_predicted = naph_model.predict(decoder_predicted_test)
##
def phenol(input):
    layer_1 = layers.Dense(units=2500, activation='elu')(input)
    layer_11 = layers.Dense(units=1000, activation='elu')(layer_1)
    layer_2 = layers.Dense(units=500, activation='elu')(layer_11)
    layer_21 = layers.Dense(units=100, activation='elu')(layer_2)
    layer_3 = layers.Dense(units=50, activation='elu')(layer_21)
    layer_4 = layers.Dense(units=10, activation='elu')(layer_3)
    output = layers.Dense(units=1, activation='elu')(layer_4)
    return output
inputs = layers.Input(shape=(np.shape(decoder_predicted_train)[1],))
phenol_output = phenol(inputs)
phenol_model = Model(inputs=inputs, outputs=phenol_output)
phenol_model.compile(optimizer='adam', loss='mean_squared_error')
deep_auto_history = phenol_model.fit(
    x=decoder_predicted_train,
    y=train_Phenol,
    epochs=300,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)
phenol_predicted = phenol_model.predict(decoder_predicted_test)
# Evaluate the results for the test data
##
mae_naph = mae(test_Naph, naph_predicted)
r2_naph = r2(test_Naph, naph_predicted)

MSE_naph = np.square(np.subtract(test_Naph, naph_predicted)).mean()
RMSE_naph = float("{:.3f}".format(np.sqrt(MSE_naph)))

##
mae_phenol = mae(test_Phenol, phenol_predicted)
r2_phenol = r2(test_Phenol, phenol_predicted)
MSE_phenol = np.square(np.subtract(test_Phenol, phenol_predicted)).mean()
RMSE_phenol = float("{:.3f}".format(np.sqrt(MSE_phenol)))
##
mae_test = [mae_naph, mae_phenol]
r2_test = [mae_naph, mae_phenol]
y = 0
##
labels = ['Naphthenic','Phenol']
concentration_predicted = [naph_predicted, phenol_predicted]
concentration_predicted = np.array(concentration_predicted)
concentration_predicted.shape = [2, test_quantity]
concentration_predicted = concentration_predicted.transpose()

real_datafrane = pd.DataFrame(test_concentration, columns=labels)
predicted_dataframe = pd.DataFrame(concentration_predicted, columns=labels)

plt.figure(figsize=(10,10))
plt.title('7 inputs')
plt.scatter(real_datafrane['Naphthenic'], predicted_dataframe['Naphthenic'], c='crimson', label="Naphthenic")
plt.scatter(real_datafrane['Phenol'], predicted_dataframe['Phenol'], c='blue', label='Phenol')

x = np.linspace(0, 100)
plt.plot(x, x, c='black', linestyle='dotted')

plt.xlabel('Measured NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=15)
plt.ylabel('Predicted NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=15)
plt.axis('square')
plt.grid(linestyle='dotted', linewidth=1)
plt.legend(fontsize=14)
plt.ylim(0, 100)
plt.xlim(0, 100)
plt.show()
##
y = 0

#predicted_dataframe.to_csv('Lake_Concentration_RF_7.csv', index=False)
predicted_dataframe.to_csv('Lake_Concentration_NN_7.csv', index=False)
y = 0


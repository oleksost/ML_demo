import json
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
import mpld3
import numpy as np
from flask import Flask
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from flask import render_template, request
import pandas as pd
app = Flask(__name__)


df_rest_p1=pd.DataFrame()
df_rest_p2=pd.DataFrame()
df_rest_p3=pd.DataFrame()

fig_person1 = plt.figure()
ax_person1 = fig_person1.add_subplot(111)
fig_person2 = plt.figure()
ax_person2 = fig_person1.add_subplot(111)
fig_person3 = plt.figure()
ax_person3 = fig_person1.add_subplot(111)


def data_preparation(df):
    d = df
    X, y = d.drop(['at_home_next'], axis=1), d['at_home_next']
    return X, y

def train_day_pred_day(X,y, person):
    X_train_1, X_train_2, y_train_1, y_train_2 = splitTrainingset(X, y, 0.5)
    # rng = np.random.RandomState(42)
    # X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, test_size=0.5, random_state=rng)
    # X_train_1, X_train_2, y_train_1, y_train_2 = np.mat(X_train_1), np.mat(X_train_2), np.array(y_train_1), np.array(y_train_2)

    # X_train = np.mat(X)
    # y_train = np.array(y)
    classifiers = [
        # ("SGD", SGDClassifier())#,
        ("SGD_perceptron", SGDClassifier(average=True, loss='perceptron')),
        ("SGD_modified_huber", SGDClassifier(average=True, loss='modified_huber')),
        ("ASGD", SGDClassifier(average=True, loss='log')),
        # ("Perceptron", Perceptron()),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0))  # ,
        # ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
    ]

    ax1 = ax_person1
    fig1 = fig_person1

    if person == "Person2":
        ax1 = ax_person2
        fig1 = fig_person2
    elif person == "Person3":
        ax1 = ax_person3
        fig1 = fig_person3


    for name, clf in classifiers:
        print "train " + name
        yy_ = []
        y_preds = np.array([])

        # Pre-training on first 50% of data
        clf = clf.fit(X_train_1, y_train_1)

        # Partial training on remaining 50% (1-sample batches)
        for i in range(X_train_2.shape[0] / 48 - 1):

            # Get train-batch
            start_ind = i * 48
            end_ind = start_ind + 47
            if end_ind > X_train_2.shape[0]:
                end_ind = X_train_2.shape[0] - 1

            x_iter_train = X_train_2[start_ind:end_ind]
            y_iter_train = y_train_2[start_ind:end_ind]

            # Train
            clf = clf.partial_fit(x_iter_train, y_iter_train, np.unique(y_train_2))

            # Get test-batch
            start_ind = (i + 1) * 48
            end_ind = start_ind + 47
            # print "Inds test: "  +str(start_ind) + " --> " + str(end_ind)
            if end_ind > X_train_2.shape[0]:
                end_ind = X_train_2.shape[0] - 1

            x_iter_test = X_train_2[start_ind:end_ind]

            # Test
            y_preds = np.append(y_preds, clf.predict(x_iter_test))
            yy_.append(1 - np.mean(y_preds == y_train_2[0:y_preds.shape[0]]))
        ax1.plot(range(len(yy_)), yy_, label=name)

    ax1.set_xlabel("Proportion train")
    ax1.set_ylabel("Test Error Rate")
    # mpld3.save_json(fig1, "abc.js")
    json01 = json.dumps(mpld3.fig_to_dict(fig1))
    return json01

#Training on 1-sample-batches, predicting next sample
def train_1_sample_batches_predict_next_sample(X,y, person):
    X_train_1, X_train_2, y_train_1, y_train_2 = splitTrainingset(X, y, 0.5)
    # rng = np.random.RandomState(42)
    # X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, test_size=0.5, random_state=rng)
    # X_train_1, X_train_2, y_train_1, y_train_2 = np.mat(X_train_1), np.mat(X_train_2), np.array(y_train_1), np.array(y_train_2)

    # X_train = np.mat(X)
    # y_train = np.array(y)
    classifiers = [
        # ("SGD", SGDClassifier())#,
        ("SGD_perceptron", SGDClassifier(average=True, loss='perceptron')),
        ("SGD_modified_huber", SGDClassifier(average=True, loss='modified_huber')),
        ("ASGD", SGDClassifier(average=True, loss='log')),
        ("Perceptron", Perceptron()),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0))
        # ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
    ]

    ax1=ax_person1
    fig1=fig_person1

    if person =="Person2":
        ax1=ax_person2
        fig1=fig_person2
    elif person=="Person3":
        ax1=ax_person3
        fig1=fig_person3

    for name, clf in classifiers:
        print "train " + name
        yy_ = []
        y_preds = np.array([])

        # Pre-training on first 50% of data
        clf = clf.fit(X_train_1, y_train_1)

        # Partial training on remaining 50% (1-sample batches)
        for i in range(X_train_2.shape[0] - 1):

            # Get train-batch
            x_iter_train = X_train_2[i]
            y_iter_train = np.array([y_train_2[i]])
            # Train
            # clf = clf.partial_fit(x_iter_train, y_iter_train, np.unique(y_train_2))

            # Get test-batch
            x_iter_test = X_train_2[i + 1]

            # Test
            y_preds = np.append(y_preds, clf.predict(x_iter_test))

            if len(y_preds) >= 1001:
                # Validate on last 1000 predictions
                yy_.append(1 - np.mean(
                    y_preds[len(y_preds) - 1001: len(y_preds) - 1] == y_train_2[len(y_preds) - 1001: len(y_preds) - 1]))

                # yy_.append(1-np.mean(y_preds == y_train_2[0:(len(y_preds))]))
        ax1.plot(range(len(yy_)), yy_, label=name)


    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)
    ax1.set_xlabel("Proportion train")
    ax1.set_ylabel("Test Error Rate")
    #mpld3.save_json(fig1, "abc.js")
    json01 = json.dumps(mpld3.fig_to_dict(fig1))
    return json01

def splitTrainingset(X,y,rate):
    amountTest = int(X.shape[0]*rate)
    X = np.mat(X)
    y = np.array(y)
    splitInd = X.shape[0]-amountTest
    maxInd = X.shape[0]
    return (X[0:splitInd], X[splitInd:maxInd], y[0:splitInd], y[splitInd:maxInd])

def splitTrainingset_for_demo(df,rate):
    amountRaws = int(df.shape[0] * rate)
    df_new, df_demo = df.ix[:amountRaws, :],  df.ix[amountRaws:, :]
    return df_new, df_demo



@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)

@app.route('/train_model', methods=['POST'])
def train_model():
    global df_rest_p1
    global df_rest_p2
    global df_rest_p3

    print "training initial..."
    person = request.form.get('person')
    df=pd.DataFrame()
    if person=='Person1':
        df = pd.read_pickle("data_3/person2_weather")
        df, df_demo=splitTrainingset_for_demo(df,0.8)
        df_rest_p1 =df_demo

    elif person=='Person2':
        df = pd.read_pickle("data_3/person3_weather")
        df, df_demo = splitTrainingset_for_demo(df, 0.8)
        df_rest_p2 = df_demo

    elif person=='Person3':
        df = pd.read_pickle("data_3/person4_weather")
        df, df_demo = splitTrainingset_for_demo(df, 0.8)
        df_rest_p3 = df_demo

    X,y=data_preparation(df)
    json_plot = train_1_sample_batches_predict_next_sample(X,y, person)
    json_data = json.dumps(json_plot)
    return json_data
    #self.emit('mpld3canvas', mpld3.fig_to_dict(fig))



@app.route('/create_new_datapoint', methods=['POST'])
def create_new_datapoint():
    global df_rest_p1
    global df_rest_p2
    global df_rest_p3

    person = request.form.get('person')
    if person == 'Person1':
        data_point = df_rest_p1.iloc[[0]]
        df_rest_p1 = df_rest_p1.ix[1:]

    elif person == 'Person2':
        data_point = df_rest_p2.iloc[[0]]
        df_rest_p1 = df_rest_p2.ix[1:]

    elif person == 'Person3':
        data_point = df_rest_p3.iloc[[0]]
        df_rest_p1 = df_rest_p3.ix[1:]

    print str(data_point.ix[0]["Longitude"])
    data2 = {}
    data2['longitude'] = data_point.ix[0]["Longitude"]
    data2['latitude'] = data_point.ix[0]["Latitude"]
    data2['weekday'] = data_point.ix[0]["weekday"]
    data2['hour'] = data_point.ix[0]["hour"]
    data2['minutes'] = data_point.ix[0]["minutes"]
    data2['apparentTemperature'] = data_point.ix[0]["apparentTemperature"]
    data2['distance_home'] = data_point.ix[0]["distance_home"]
    data2['downfall'] = data_point.ix[0]["downfall"]
    data2['time'] = str(data_point.index[0])
    json_data = json.dumps(data2)
    #print str(data2)
    return json_data






if __name__ == '__main__':
   app.run()
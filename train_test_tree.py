import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from utils import *

class CutEngine():
    def __init__(self, df, train_col, lr = 1):
        self.df = df
        self.features = train_col[:-1]
        # Remove unused columns
        df = df[train_col]

        # Split the data into train, validation, and test sets
        X = df.drop(columns=['sig'])
        y = df['sig']

        # X["pmu"] = np.log10(1e-12 + X["pmu"])
        # X["ppi0"] = np.log10(1 - X["ppi0"] + 1e-12)
        X_train, self.X_v, self.y_train, y_v = train_test_split(X, y, test_size=0.2, random_state = 42)
        # plot_class_frac(X_train)
        # plot_class_frac(self.X_v)
        # X_train = X_train[train_col]
        # self.X_v = X_train[train_col]
        X_test, X_cal, self.y_test, self.y_cal = train_test_split(self.X_v, y_v, test_size = 0.5, random_state = 42)
        X_test = self.X_v
        self.y_test = y_v
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        # scaler = PowerTransformer()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.X_cal_scaled = scaler.transform(X_cal)
        self.X_scaled = scaler.transform(X)
        # Calculate class weights
        class_weights = {0: 1, 1: 1.7}  # Assigning a weight of 1 to class 0 and 1000 to class 1

        # self.rf_model = RandomForestClassifier(class_weight = class_weights)
        # self.rf_model = SVC(class_weight = class_weights)
        print("using gbdt model")
        self.rf_model = GradientBoostingClassifier(n_estimators = 150, subsample = 0.5, max_features = len(train_col)-1, tol = 1e-6)
        class_weights = {0:1, 1: 1.8}
        # c_weight = [1.7,1.8,1.9,2]
        # g = "scale"
        # k = "rbf"
        # self.rf_model = SVC(class_weight = class_weights, gamma = g, kernel = k, C = 5.6, probability = True)
            
        gbdt = GradientBoostingClassifier(n_estimators = 200, subsample = 0.5, max_features = len(train_col)-1)
        # self.rf_model = AdaBoostClassifier()
        # ada = AdaBoostClassifier()
        self.rf_model_cal = CalibratedClassifierCV(gbdt)
        

    def train(self):
        # Train the chosen classifier
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        self.rf_model_cal.fit(self.X_train_scaled, self.y_train)

    def plot_probs(self, test_prob):
        plt.hist([test_prob[self.y_test == i] for i in range(2)], histtype = "barstacked", bins = 50, range = (0.1,1), label = ["Background", "Signal"])
        plt.xlabel("Computed signal probability")
        plt.ylabel("Number of events")
        plt.legend()
        plt.show()
        plt.clf()

    def test(self, weight_rec = 1.3):
        # Evaluate on the test set
        test_prob = self.rf_model.predict_proba(self.X_test_scaled)[:,1]
        best_f1 = 0
        self.best_thresh = 0
        best_pr = (0,0)
        for threshold in np.linspace(0.1,0.8,40):
            test_predictions = (test_prob>=threshold).astype(int)
            test_accuracy = accuracy_score(self.y_test, test_predictions)
            test_f1, prec, rec = self.weighted_f1(self.y_test, test_predictions, w = weight_rec)
            if test_f1 > best_f1: 
                best_f1 = test_f1
                self.best_thresh = threshold
                best_pr = (prec,rec)
        test_predictions = (test_prob>=self.best_thresh).astype(int)
        test_f1, prec, rec = self.weighted_f1(self.y_test, test_predictions)
        cm = confusion_matrix(self.y_test, test_predictions)
        return test_f1, prec, rec
    
    def test_on_train(self):
        # Evaluate on the test set
        test_prob = self.rf_model.predict_proba(self.X_train_scaled)[:,1]
        plt.hist([test_prob[self.y_train == i] for i in range(2)], histtype = "barstacked", bins = 50)
        plt.show()
        # Adjust the decision threshold
        desired_false_negative_rate = 0.4  # Set your desired false negative rate here

        # Find the appropriate threshold to achieve the desired false negative rate
        # threshold = sorted(test_prob)[int((1 - desired_false_negative_rate) * len(test_prob))]
        # print(threshold)
        for threshold in np.linspace(0.4,0.6,20):
            test_predictions = (test_prob>=threshold).astype(int)
            test_accuracy = accuracy_score(self.y_train, test_predictions)
            test_f1 = self.weighted_f1(self.y_train, test_predictions)
            cm = confusion_matrix(self.y_train, test_predictions)
        return test_f1
    
    def make_calibration_curve(self):
        n_sample = len(self.y_cal[self.y_cal == 1])
        frac_list = np.linspace(0,1,10)
        for (name, model) in [("base model",self.rf_model), ("calibrated model",self.rf_model_cal)]:
            cal_pred = model.predict_proba(self.X_cal_scaled)[:,1]
            pos_prob = cal_pred[self.y_cal==1]
            neg_prob = cal_pred[self.y_cal==0]
            prob_avg_list = []
            for frac in frac_list : 
                prob_sel = np.concatenate((np.random.choice(pos_prob, np.int(frac*n_sample)),np.random.choice(neg_prob, n_sample - np.int(frac*n_sample))))
                prob_avg_list.append(np.mean(prob_sel))
            plt.plot(prob_avg_list, frac_list, label = name)

        plt.plot(frac_list, frac_list, label = 'Perfectly calibrated', color = "b", linestyle = "dashed")
        plt.xlabel("Mean predicted signal probability")
        plt.ylabel("Signal sample fraction")
        plt.legend()
        plt.show()

    def get_pred_labels(self):
        data_prob = self.rf_model.predict_proba(self.X_scaled)[:,1]
        preds = (data_prob>=self.best_thresh).astype(int)
        return preds

    def get_features_importances(self):
        imp = self.rf_model.feature_importances_
        dic_imp = {}
        for i in range(len(self.features)):
            dic_imp[self.features[i]] = imp[i]
        return dic_imp

    def optimize(self):
        c_list = [5, 5.2, 5.4, 5.6, 5.8,6,6.2,6.5]
        # kernel_list = ["linear", "rbf", "sigmoid"]
        kernel_list = ["sigmoid", "rbf", "linear", "poly"]
        gamma_list = ["scale"]
        c_weight = [1.7,1.8,1.9,2]
        self.best_f1 = 0
        g = "scale"
        k = "rbf"
        for x in c_list : 
            for c in c_weight:
                class_weights = {0:1, 1: c}
                self.rf_model = SVC(class_weight = class_weights, gamma = g, kernel = k, C = x)
                self.train()
                f1 = self.test()
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    print("best vals : ", x,k,g,c)

    def weighted_f1(self, y_true, y_pred, w= 0.2):
        #gives more importance to recall over precision in F1 score
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        mod_f1 = (1+w)*(prec*rec)/(w*prec + rec)
        return mod_f1, prec, rec


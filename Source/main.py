import numpy as np

from fcm_estimator import FCMeansEstimator
from ga import Genetic_Algorithm
from svr_estimator import SVREstimator
import pandas as pd
import os
import sys
import time


def compute_rmse(a, b):
    return np.sqrt(np.square(np.subtract(a, b)).sum().sum())/np.sqrt(np.square(a).sum().sum())


def fillMissingValues(incomplete_data, imputedData, incomplete_rows):

    for i in range(len(incomplete_rows)):
        incomplete_data.iloc[incomplete_rows.iloc[i]
                             ] = incomplete_data.iloc[incomplete_rows.iloc[i]].fillna(imputedData.iloc[i])

    return incomplete_data


def load_data(file_name):
    data_split = file_name.split("_")
    dataset_no = data_split[1]
    incomplete_data = pd.read_csv(os.path.join(os.getcwd(
    ), "..\\Incomplete datasets\Data "+dataset_no+"\\"+file_name), header=None)
    complete_data = pd.read_csv(os.path.join(
        os.getcwd(), "..\\Complete datasets\Data_"+dataset_no+".csv"), sep=',')

    return np.array(complete_data), np.array(incomplete_data)
# Run the algorithm for estimating


def main(file_name):

    complete_data, incomplete_data = load_data(file_name)
    #data = np.array(pd.read_csv(r"C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Complete datasets\\Data_3.csv", sep=','))
    # Make FCM and SVR model

    #path_src = 'C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Incomplete datasets\\Data 3'
    # for i in os.listdir(path_src):

    # if i.startswith('.'):
    # continue
    print("Started missing imputation for:", file_name)
    #incomplete_data = np.array(pd.read_csv(path_src+'/'+i, sep=',' , header=None))

    st = time.time()
    svr_estimator = SVREstimator(data=incomplete_data)
    #y = svr_estimator.estimate_missing_value()
    ga = Genetic_Algorithm(svr_estimator, incomplete_data)

    while True:
        c, m = ga.run()
        fcm_estimator = FCMeansEstimator(c=c, m=m, data=incomplete_data)
        x = fcm_estimator.estimate_missing_values()
        #error = np.power(x - y, 2).sum()
        imputedData = fillMissingValues(pd.DataFrame(incomplete_data), pd.DataFrame(x), pd.DataFrame(svr_estimator.incomplete_rows))
        # incomplete_data.fillna()
        rmse = compute_rmse(complete_data, imputedData)

        print('RMSE  : ' + str(round(rmse, 4)))
        print("C:", c, "M:", m)
        et = time.time()
        time_taken = round(et-st, 2)
        print('Execution time:', time_taken, 'seconds')
        print("------------------------------------------------")
        '''
        original_stdout = sys.stdout with open('C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Output_3.txt', 'a') as f:
        sys.stdout = f

        print(i)
        print('RMSE  : ' + str(round(rmse, 4)))
        print("C:", c, "M:", m)
        print('Execution time:', time_taken, 'seconds')
        print("------------------------------------------------")
        sys.stdout = original_stdout
        '''
        if(rmse < 1):
            break


if __name__ == '__main__':
    # Use the name of the database as input
    main("Data_2_NN_5%.csv")

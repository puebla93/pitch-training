import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def main():
    x_test, y_test, z_test, x_result, y_result, z_result = get_cordenates()

    # graphics(x_test, y_test, z_test, x_result, y_result, z_result)

    get_error(x_test, y_test, z_test, x_result, y_result, z_result)

def get_balls():
    test_path = 'data_set/fullHD/test/'
    tests_files = os.listdir(test_path)
    tests_files.sort()

    result_path = 'data_set/fullHD/results/'
    results_files = os.listdir(result_path)
    results_files.sort()

    test_number = 1
    test_fd = open(test_path + tests_files[test_number])
    test_data = json.load(test_fd)

    result_fd = open(result_path + results_files[test_number])
    result_data = json.load(result_fd)

    test_balls = test_data.values()
    test_balls.sort(key=lambda ball: ball[0][0])
    result_balls = map(lambda key: result_data[key], test_data.keys())
    result_balls.sort(key=lambda ball: ball[0][0])
    return test_balls, result_balls

def get_cordenates():
    test_balls, result_balls = get_balls()

    x_test = np.array(map(lambda ball: ball[0][0], test_balls))
    # x_test.sort()
    y_test = np.array(map(lambda ball: ball[0][1], test_balls))
    # y_test.sort()
    z_test = np.array(map(lambda ball: ball[1], test_balls))
    # z_test.sort()

    x_result = np.array(map(lambda ball: ball[0][0], result_balls))
    # x_result.sort()
    y_result = np.array(map(lambda ball: ball[0][1], result_balls))
    # y_result.sort()
    z_result = np.array(map(lambda ball: ball[1], result_balls))
    # z_result.sort()

    return x_test, y_test, z_test, x_result, y_result, z_result

def graphics(x_test, y_test, z_test, x_result, y_result, z_result):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_test, y_test, z_test, c='g', label="valores reales")
    ax.plot(x_result, y_result, z_result, c='r', label="valores del sistema")
    ax.legend()

    # plt.plot(x_test, y_test, c='g', label="valores reales")
    # plt.plot(x_result, y_result, c='r', label="valores del sistema")
    # plt.legend()

    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(0, 1)
    # ax.set_zlim3d(0, 1)

    plt.show()

def get_error(x_test, y_test, z_test, x_result, y_result, z_result):
    try:
        fd = open("data_set/fullHD/ECM.json")
        data = json.load(fd)
    except IOError:
        data = {}
    n = len(x_test)
    distances = ((x_result - x_test)**2 + (y_result - y_test)**2 + (z_result - z_test)**2)**.5
    err_per_point = np.sum(distances)/n
    data["pitch2"] = [err_per_point, 3.45*err_per_point/320]
    try:
        fd = open("data_set/fullHD/ECM.json", 'w')
        json.dump(data, fd, indent=4)
    except UnicodeDecodeError:
        pass
    # return err_per_point

if __name__ == '__main__':
    main()

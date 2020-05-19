import os
import numpy as np
import cv2
import detect_homes
import transform
import capture_balls
from cvinput import cvwindows
from parse_args import args
from util.utils import Reader, Obj, HomePlate, Ball
from util.utils import show_contours, homeAVG, kmeans, draw_finalResult, plot_fit, draw_strikeZone, fit_velocity, plot_velocity, draw_SideTrajectory
from filtering import filter_img
import ransac
import get_results

params = Obj(
    useKmeans=False,
    kmeans_k=6,
    transform_resolution=(600, 1024),
    camera_fps=187.,
    home_large=43.18,
    ball_diameter=7.2644,
    camera_hight=225,
    strike_zone_up = 107.8738,
    strike_zone_down = 39.4462
)

def main2():
    from pitch_training import PitchTrainig
    pitchTrainig = PitchTrainig()
    while cvwindows.event_loop():
        home = pitchTrainig.calibrateHome()

        PTM, user_homePlate_cnt = pitchTrainig.computeTransform(home)

        pitchTrainig.waitBalls(PTM)

def main():
    reader = Reader()
    setUp_Reader(reader)

    # setting up transform, detect_homes and capture_balls params
    detect_homes.setUp({"debugging":args.debugging})
    transform.setUp({"debugging":args.debugging})
    capture_balls.setUp({"debugging":args.debugging})
    ransac.setUp({"debugging":args.debugging})

    home = calibrateHome(reader)

    PTM, new_homePlate = computeTransform(reader, home)

    capture_balls.setUp({"home_begin":new_homePlate[2,0]})
    balls_tracked = waitBalls(reader, PTM)

    balls, model = fit_balls(balls_tracked)

    wasStrike = was_strike(new_homePlate, model)

    points = get_wordPoints(balls, new_homePlate)
    velocity = get_velocity(points)

    # draw final result
    # user_img = draw_finalResult(new_homePlate, balls, params.transform_resolution, wasStrike, velocity)
    user_img = draw_strikeZone((int(225*2.31), 600), model, wasStrike, velocity)
    # user_img = draw_SideTrajectory((int(225*2.31), 1024), balls, model[1], wasStrike, velocity)
    cv2.imshow('RESULT', user_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def calibrateHome(reader):
    home_tracking = []
    # while len(home_tracking) < 90:
    while reader._frameNumber < 90:
        # reading a frame
        frame = reader.read()
        if frame is None:
            break

        # removing noise from image
        blur = filter_img(frame)

        # using kmeans on the image
        if params.useKmeans:
            blur = kmeans(blur, params.kmeans_k)
            if args.debugging:
                cv2.imshow('kmeans', blur)
                cv2.waitKey(0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # finding a list of homes
        homes = detect_homes.get_homes(gray)
        if homes is None or len(homes) == 0:
            continue

        # keep the best home
        home = homeAVG(homes)
        home_tracking.append(home)

        contours_img = frame.copy()
        cv2.drawContours(contours_img, [home.contour.astype('int32')], -1, (0, 0, 255), 2)
        cv2.imshow('Homes', contours_img)
        cv2.waitKey(1)

        if len(homes) > 2:
            print("len = ", len(homes))
            print(reader.get_frameName())
            cv2.waitKey(0)

    cv2.destroyWindow('Homes')
    return homeAVG(home_tracking)

def computeTransform(reader, home):
    gray = filter_img(reader.actualFrame)
    PTM, new_homePlate_cnt = transform.homePlate_transform(gray, home)
    return PTM, new_homePlate_cnt

def waitBalls(reader, PTM):
    balls_tracked = []
    frame_number = 0
    # loop
    while True:
        # reading a frame
        frame = reader.read()
        if frame is None:
            break

        # removing noise from image
        blur = filter_img(frame)

        # using kmeans on the image
        if params.useKmeans:
            blur = kmeans(frame, params.kmeans_k)
            if args.debugging:
                cv2.imshow('kmeans', blur)
                cv2.waitKey(0)
        
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # transform the frame
        # warped = cv2.warpPerspective(frame, PTM, transform.params.transform_resolution)
        warped = cv2.warpPerspective(gray, PTM, transform.params.transform_resolution)

        # finding the ball
        balls = capture_balls.get_balls(warped, frame_number)
        if balls.shape[0] > 0:
            balls_tracked.append(balls)

            if args.save_as_test:
                warped1 = cv2.warpPerspective(frame, PTM, transform.params.transform_resolution)
                get_results.saveBallPosition(warped1, reader.get_frameName())

                folder_path = os.listdir("pelota")
                folder_path.sort()
                folder_name = folder_path[args.test_folder]
                file_path = 'data_set/PSEye/results/' + folder_name + ".json"
                data = get_results.load(file_path)
                data[reader.get_frameName()] = [list(balls[0].center), balls[0].radius]
                get_results.save(data, file_path)            

        cv2.imshow('camera', frame)
        cv2.waitKey(1)
        frame_number += 1

    cv2.destroyWindow('camera')
    return np.array(balls_tracked)

def fit_balls(balls_tracked):
    all_balls = np.array([ball for balls in balls_tracked for ball in balls])
    points = [np.array([ball.center[0], ball.center[1], ball.radius, ball.capture_frame]) for ball in all_balls]
    points = np.array(points)
    new_points, model = ransac.ransac(points)
    balls = [Ball(np.array(point[:2]), point[2], point[3]) for point in new_points]

    if args.debugging:
        plot_fit(all_balls, balls)
    
    return balls, model

def was_strike(homePlate, ballFunc):
    YballFunc, ZballFunc = ballFunc
    Yfunc = lambda x: YballFunc[0] + YballFunc[1]*x + YballFunc[2]*x**2
    Zfunc = lambda x: ZballFunc[0] + ZballFunc[1]*x + ZballFunc[2]*x**2

    start, stop = int(homePlate[2, 0]), (int(homePlate[1, 0]) + 1)
    range1, range2 = homePlate[3, 1], homePlate[2, 1]
    home_large_pixels = abs(range2 - range1)
    ball_diameter_pixels = params.ball_diameter / params.home_large * home_large_pixels
    for x in range(start, stop):
        y = Yfunc(x)
        if y >= range1 and y <= range2:
            ball_pixels = Zfunc(x)*2.
            ball_high = params.camera_hight - (params.camera_hight * ball_diameter_pixels / ball_pixels)
            if ball_high >= params.strike_zone_down and ball_high <= params.strike_zone_up:
                return True
    return False

def get_wordPoints(balls, homePlate):
    home_large_pixels = abs(homePlate[2, 1] - homePlate[3, 1])
    ball_diameter_pixels = params.ball_diameter / params.home_large * home_large_pixels
    points = []
    for ball in balls:
        ball_high = params.camera_hight - (params.camera_hight * ball_diameter_pixels / (ball.radius*2.))
        cm_per_pixel = params.ball_diameter/(ball.radius*2.)
        new_point = [ball.center[0] * cm_per_pixel, ball.center[1] * cm_per_pixel, ball_high, ball.capture_frame]
        points.append(np.array(new_point))
    points = np.array(points)
    return points

def get_velocity(points):
    velocity_per_point = []
    frame_numbers = []
    time_between_frames = 1./params.camera_fps*0.0002778 # convert time_between_frames in seconds to hours
    cm_to_mile = 6.2137119e-6
    i = 0
    while i < len(points)-1:
        cm_dist = ((points[i][0]-points[i+1][0])**2+(points[i][1]-points[i+1][1])**2+(points[i][2]-points[i+1][2])**2)**.5
        # cm_dist = ((points[i][0]-points[i+1][0])**2)**.5
        
        mile_dist = cm_dist*cm_to_mile
        time = abs(points[i][3]-points[i+1][3])*time_between_frames
        if time == 0:
            i += 1
            continue

        velocity_point = mile_dist/time
        velocity_per_point.append(velocity_point)
        frame_numbers.append(points[i+1,3])

        i += 1

    data_list = list(map(lambda f, v: np.array([f, v]), frame_numbers, velocity_per_point))
    data = np.array(data_list)
    func = fit_velocity(data)
    ball_velocity = func(points[-1,3])

    if args.debugging:
        plot_velocity(data, func)

    return str(int(ball_velocity)) + " MPH"

def setUp_Reader(reader):
    test_cases_path = "test_cases/ps_eye (320x240_187fps)"
    folder_path = os.listdir(test_cases_path)
    folder_path.sort()
    path = test_cases_path+ '/' + folder_path[args.test_folder] + '/'
    framesNames = os.listdir(path)
    framesNames.sort(key=lambda frameName: int(frameName[:-4]))
    reader._frameNumber = args.test_frame - 1
    reader_params = {}
    reader_params["folder_path"] = path
    reader_params["frameNames"] = framesNames
    reader.setUp(reader_params)

def setUp(nparams):
    params.setattr(nparams)

if __name__ == "__main__":
    main()

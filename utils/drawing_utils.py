def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)


def calculate_speed(position1, position2, time_interval):
    """ Calculate speed given two positions and time interval """
    distance = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
    speed = distance / time_interval
    return speed
import csv
import re
import json
from line import *
from point import *
from path import *

def read_picture_names_and_descriptions(js_file_name):
    #Parse the json file, store picture names and descriptions.
    names = {}
    descriptions = {}

    # Load the file
    with open(js_file_name) as f:
        data = json.load(f)

    for element in data:
        pic_id = str(element["id"])
        pic_name = element["name"]
        pic_desc = element["description"]

        names[pic_id] = pic_name
        descriptions[pic_id] = pic_desc


    return names, descriptions

def read_picture_names_and_descriptions_17(js_file_name):
    #Parse the json file, store picture names and descriptions.
    names = {}
    descriptions = {}

    # Load the file
    with open(js_file_name) as f:
        data = json.load(f)

    for element in data["atlas"]:
        pic_id = str(element["id"])
        pic_name = element["name"]
        pic_desc = element["description"]

        names[pic_id] = pic_name
        descriptions[pic_id] = pic_desc


    return names, descriptions


def store_locations(js_filename,proj=None):
    # Parse the json file, store the Path objects of every image within the canvas, and return as a dictionary indexed by the picture id.
    locations = dict()

    # Load the file
    with open(js_filename) as f:
        data = json.load(f)

    for element in data["atlas"]:
        pic_id = str(element["id"])

        if proj is None or proj == pic_id:
            name = element["name"]

            path = Path(pic_id, name)
            points = element["path"]

            if len(points) > 0:
                # The first point in points is also the ending point, so add a copy of it to the end
                first_element = points[0]
                points.append(first_element)
                for i in range(len(points) - 1):
                    start_x = points[i][0]
                    start_y = points[i][1]
                    end_x = points[i+1][0]
                    end_y = points[i+1][1]

                    point1 = Point(start_x, start_y)
                    point2 = Point(end_x, end_y)
                    line = Line(point1, point2)
                    path.add_line(line)

                locations[pic_id] = path

    return locations

def store_locations_22(js_filename,proj=None):
    # Parse the json file, store the Path objects of every image within the canvas, and return as a dictionary indexed by the picture id.
    locations = dict()

    # Load the file
    with open(js_filename) as f:
        data = json.load(f)

    for element in data:
        pic_id = str(element["id"])

        if proj is None or proj == pic_id:
            name = element["name"]

            path = Path(pic_id, name)
            points = []
            for pa in element["path"]:
                points = element["path"][pa]

            if len(points) > 0:
                # The first point in points is also the ending point, so add a copy of it to the end
                first_element = points[0]
                points.append(first_element)
                for i in range(len(points) - 1):
                    start_x = points[i][0]
                    start_y = points[i][1]
                    end_x = points[i+1][0]
                    end_y = points[i+1][1]

                    point1 = Point(start_x, start_y)
                    point2 = Point(end_x, end_y)
                    line = Line(point1, point2)
                    path.add_line(line)

                locations[pic_id] = path
            else:
                print("length is 0" + pic_id)
    return locations

def store_locations_23(js_filename,proj=None):
    # Parse the json file, store the Path objects of every image within the canvas, and return as a dictionary indexed by the picture id.
    locations = dict()

    # Load the file
    with open(js_filename) as f:
        data = json.load(f)

    for element in data:
        pic_id = str(element["id"])

        if proj is None or proj == pic_id:
            name = element["name"]

            path = Path(pic_id, name)
            points = []
            for pa in element["path"]:
                points = element["path"][pa]

            if len(points) > 0:
                # The first point in points is also the ending point, so add a copy of it to the end
                first_element = points[0]
                points.append(first_element)
                for i in range(len(points) - 1):
                    start_x = points[i][0] + 1500
                    start_y = points[i][1] + 1000
                    end_x = points[i+1][0] + 1500
                    end_y = points[i+1][1] + 1000

                    point1 = Point(start_x, start_y)
                    point2 = Point(end_x, end_y)
                    line = Line(point1, point2)
                    path.add_line(line)

                locations[pic_id] = path
            else:
                print("length is 0" + pic_id)
    return locations


if __name__ == "__main__":
	locations = store_locations_22("../data/atlas.json")

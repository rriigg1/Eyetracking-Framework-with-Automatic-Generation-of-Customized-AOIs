import config


def calibrate_cross(data, columns=config.ET_FILE_HEADERS["default"]):
    """
    Calculates by how much the data is shifted and corrects wrong screen dimesnions.
    """
    if ("fixation_duration" not in columns):
        raise ValueError("Fixation duration needed for calibration.")

    longest_fixation = 0
    center = {"x": config.SCREEN_SIZE["width"]/2, "y": config.SCREEN_SIZE["height"]/2}
    fixcross = {"x": config.SCREEN_SIZE["width"]/2, "y": config.SCREEN_SIZE["height"]/2}
    for row in data:
        cur_dur = float(row[columns["fixation_duration"]])
        if (longest_fixation < cur_dur):
            longest_fixation = cur_dur
            fixcross["x"] = float(str(row[columns["fixation_x"]]).replace(",", "."))
            fixcross["y"] = float(str(row[columns["fixation_y"]]).replace(",", "."))
    return {"x": center["x"]-fixcross["x"],
            "y": center["y"]-fixcross["y"],
            "scale_x": config.VIDEO_SIZE["width"]/config.SCREEN_SIZE["width"],
            "scale_y": config.VIDEO_SIZE["height"]/config.SCREEN_SIZE["height"]}


def shift_correct(coordinate, c_data):
    """
    Corrects for shift and scaling errors in the data.
    """
    x, y = coordinate
    x = (x + c_data["x"]) * c_data["scale_x"]
    y = (y + c_data["y"]) * c_data["scale_y"]
    return (x, y)

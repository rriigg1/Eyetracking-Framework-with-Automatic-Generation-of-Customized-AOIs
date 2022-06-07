import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import os
from EyetrackingFramework.config import AOI_GROUPINGS
from EyetrackingFramework.config import DISTANCE_FUNCTIONS
from EyetrackingFramework import landmark_utils
from EyetrackingFramework import visualizations
from EyetrackingFramework import aoi
from EyetrackingFramework import config as config


GROUPINGS_NAME = "AOI_GROUPINGS"
DEFAULT_FACE = None
UNDIRECTIONAL = ["POINT", "HULL", "VORONOI-HESSELS"]
FACE_IMAGE = None
im_size = config.DEFAULT_LANDMARK_SIZE
LANDMARKS = landmark_utils.get_default_landmarks()
SCALED_LANDMARKS = list(map(lambda p: (p[0]/im_size[1], p[1]/im_size[0]), LANDMARKS))

GROUPING_COLUMN = 0
CANVAS_COLUMN = 1
PARTS_COLUMN = 5

FILE_MENU_ROW = 0
BOTTOM_ROW = 7


def set_background_image(image):
    """
    Sets the background image. The given image must contain a face.
    """
    global FACE_IMAGE, LANDMARKS, im_size, SCALED_LANDMARKS
    newLandmarks = landmark_utils.get_landmarks(image)
    if (newLandmarks is None):
        messagebox.showwarning(title="No face in image.", message="No face was detected in the given image.")
        return False
    FACE_IMAGE = image
    LANDMARKS = newLandmarks
    im_size = FACE_IMAGE.shape
    SCALED_LANDMARKS = list(map(lambda p: (p[0]/im_size[1], p[1]/im_size[0]), LANDMARKS))
    return True


def find_dict_multiline_definition(lines, name):
    """
    Finds the index in lines where a dictionary with the given name is defined.
    """
    for li, l in enumerate(lines):
        i = l.find(name)
        if (0 <= i < l.find("=")):
            if (l.count("{") > l.count("}")):
                # this should be a valid defintion of the wanted dictionary spanning multiple lines
                return li
    return -1


def list_to_ranges(numbers):
    """
    Converts a list of numbers to a concatenation of consecutive numbers
    expressed as range statements.
    The ranges are of maximum length.
    The result is a string that contains python code that produces the given list.
    ###Example:
        [1,2,3,4,5,6] => list(range(1,7))
    """
    text = ""
    start = 0
    end = 0
    cur_list = []
    if (len(numbers) <= 0):
        return "[]"
    while end < len(numbers):
        if (len(numbers) > end+1 and numbers[end+1] == numbers[end]+1):
            end += 1
        else:
            if (end - start + 1 <= 3):
                cur_list += numbers[start:end+1]
            else:
                if (len(cur_list) >= 1):
                    if (text != ""):
                        text += " + "
                    text += "["
                    for i, n in enumerate(cur_list):
                        text += str(n)
                        if (i < len(cur_list)-1):
                            text += ", "
                    text += "]"
                if (numbers[start] > 0):
                    text += f"list(range({numbers[start]}, {numbers[end]+1}))"
                else:
                    text += f"list(range({numbers[end]+1}))"
            start = end+1
            end = start
    if (len(cur_list) >= 1):
        if (text != ""):
            text += " + "
        text += "["
        for i, n in enumerate(cur_list):
            text += str(n)
            if (i < len(cur_list)-1):
                text += ", "
        text += "]"
    return text


def write_groupings(aoi_groupings, file_name="test_config.py"):
    """
    Replaces the dictionary that contains the groupings in a selected config file
    with the given new groupings dictionary.
    """
    if (not os.path.isfile(file_name)):
        print(f"\'{file_name} is not a valid name.\'")
    read_file = file_name
    if (not os.path.exists(file_name)):
        read_file = config.__file__
    with open(read_file, "r") as config_file:
        lines = config_file.readlines()
    if (lines is None):
        print("Failed to open config file.")
        return False
    start = find_dict_multiline_definition(lines, GROUPINGS_NAME)
    if (start < 0):
        print(f"Could not find groupings: \'{GROUPINGS_NAME}\'")
        return False
    i = start
    diff = lines[start].count("{")-lines[start].count("}")
    while diff > 0:
        i += 1
        if (i >= len(lines)):
            i = -1
            break
        diff += lines[i].count("{")-lines[i].count("}")
    if (diff != 0):
        print("Brackets of AOI_GROUPINGS are not balanced.")
    end = i
    prefix = lines[:start]
    suffix = lines[end+1:]
    aoi_text = GROUPINGS_NAME + " = {\n"
    for name, grouping in aoi_groupings.items():
        aoi_text += f"\t\"{name}\":\n"
        aoi_text += "\t\t{\n"
        if (len(grouping) <= 0):
            aoi_text = aoi_text[:-1] + "},\n"
            continue
        for n, parts in grouping.items():
            aoi_text += f"\t\t\t\"{n}\": ["
            for pi, p in enumerate(parts):
                aoi_text += "("
                aoi_text += list_to_ranges(p[0])
                aoi_text += f", \"{p[1]}\", {p[2]})"
                if (pi < len(parts)-1):
                    aoi_text += ", "
            aoi_text += "],\n"
        aoi_text = aoi_text[:-2]
        aoi_text += "\n\t\t},\n"
    aoi_text = aoi_text[:-2]
    aoi_text += "\n}\n"
    aoi_text = aoi_text.replace("\t", "    ")
    with open(file_name, "w") as config_file:
        config_file.writelines(prefix)
        config_file.write(aoi_text)
        config_file.writelines(suffix)
    return True


class AOI_editor:
    """
    GUI for showing and editing groupings and their AOIs.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.minsize(500, 500)
        self.window.title("AOI creator")
        self.create_widgets()
        self.window.update()
        self.redraw()
        self.group_list = {}

    def create_widgets(self):
        """
        Create all widgets. This is function is only called once when creating the window.
        """
        self.window.columnconfigure(CANVAS_COLUMN+1, weight=1)
        self.window.columnconfigure(GROUPING_COLUMN, minsize=200)
        self.window.rowconfigure(1, weight=5)
        self.window.rowconfigure(6, weight=1)
        self.window.bind("<KeyPress>", self.__window_key)
        # flags used to determine which widgets to draw
        self.selected_grouping = None
        self.selected_aoi = None
        self.selected_part = None
        self.selected_landmark = None
        self.selected_lm_idx = None
        # groupings
        self.grouping_lbl = tk.Label(self.window, text="Groupings")
        self.grouping_lbl.grid(row=0, column=GROUPING_COLUMN)
        self.grouping_box = tk.Listbox(self.window, exportselection=0)
        self.grouping_box.grid(row=1, column=GROUPING_COLUMN, rowspan=6, sticky="nswe")
        self.grouping_box.bind("<<ListboxSelect>>", self.__group_selected)
        self.add_grouping_btn = tk.Button(self.window, text="Add grouping", command=self.new_grouping)
        self.add_grouping_btn.grid(row=BOTTOM_ROW, column=GROUPING_COLUMN, sticky="we")
        self.remove_grouping_btn = tk.Button(self.window, text="Remove grouping", command=self.remove_grouping)
        self.remove_grouping_btn.grid(row=BOTTOM_ROW+1, column=GROUPING_COLUMN, sticky="we")
        self.rename_grouping_btn = tk.Button(self.window, text="Rename grouping", command=self.rename_grouping)
        self.rename_grouping_btn.grid(row=BOTTOM_ROW+2, column=GROUPING_COLUMN, sticky="we")
        # canvas
        self.draw_canvas = tk.Canvas(self.window, bg="white")
        self.draw_canvas.grid(row=1, column=CANVAS_COLUMN, rowspan=6, columnspan=4, sticky="nswe")
        self.draw_canvas.bind("<Configure>", self.redraw)
        self.draw_canvas.bind("<ButtonRelease-1>", self.__canvas_click)
        self.draw_canvas.bind("<KeyPress>", self.__canvas_key)
        self.canvas_size = (self.draw_canvas.winfo_width(), self.draw_canvas.winfo_height())
        self.aoi_image = Image.fromarray(np.full((self.canvas_size[1], self.canvas_size[0], 3), 255, np.uint8))
        # file menu
        #self.config_file = "test_config.py"
        #self.config_entry = tk.Entry(self.window)
        #self.config_entry.insert(0, self.config_file)
        #self.config_entry.grid(row=FILE_MENU_ROW, column=CANVAS_COLUMN, columnspan=2, sticky="we")
        #self.load_config_btn = tk.Button(self.window, text="...", command=self.get_file)
        #self.load_config_btn.grid(row=FILE_MENU_ROW, column=CANVAS_COLUMN+2, sticky="we")
        self.save_config_btn = tk.Button(self.window, text="Save groupings", command=self.save_groupings)
        self.save_config_btn.grid(row=FILE_MENU_ROW, column=CANVAS_COLUMN, sticky="we")
        self.background_menu = tk.Menu(self.window, title="Test", tearoff=0)
        self.background_menu.add_command(label="Load image", command=self.__load_background)
        self.background_menu.add_command(label="Reset scale", command=self.__rescale_background)
        self.draw_canvas.bind("<ButtonRelease-3>", self.__background_popup)
        self.background_btn = tk.Button(self.window, text="Background Image", command=self.__load_background)
        self.background_btn.grid(row=FILE_MENU_ROW, column=CANVAS_COLUMN+1, sticky="e")
        # aois
        self.aoilist_lbl = tk.Label(self.window, text="AOIs")
        self.aoilist_lbl.grid(row=0, column=PARTS_COLUMN)
        self.aoi_listbox = tk.Listbox(self.window, exportselection=0)
        self.aoi_listbox.grid(row=1, column=PARTS_COLUMN, sticky="nswe")
        self.aoi_listbox.bind("<<ListboxSelect>>", self.__aoi_selected)
        self.add_aoi_btn = tk.Button(self.window, text="Add AOI", command=self.new_aoi)
        self.add_aoi_btn.grid(row=2, column=PARTS_COLUMN, sticky="we")
        self.remove_aoi_btn = tk.Button(self.window, text="Remove AOI", command=self.remove_aoi)
        self.remove_aoi_btn.grid(row=3, column=PARTS_COLUMN, sticky="we")
        self.rename_aoi_btn = tk.Button(self.window, text="Rename AOI", command=self.rename_aoi)
        self.rename_aoi_btn.grid(row=4, column=PARTS_COLUMN, sticky="we")
        self.aoi_sel_index = None
        self.aoi_listbox.bind("<Button-1>", self.__start_aoi_drag)
        self.aoi_listbox.bind("<ButtonRelease-1>", self.__end_aoi_drag)
        self.aoi_listbox.bind("<Leave>", self.__end_aoi_drag)
        # aoi parts
        self.partlist_lbl = tk.Label(self.window, text="Parts")
        self.partlist_lbl.grid(row=5, column=PARTS_COLUMN)
        self.parts_listbox = tk.Listbox(self.window, exportselection=0)
        self.parts_listbox.grid(row=6, column=PARTS_COLUMN, rowspan=2, sticky="nswe")
        self.parts_listbox.bind("<<ListboxSelect>>", self.__part_selected)
        self.add_part_btn = tk.Button(self.window, text="Add Part", command=self.new_part)
        self.add_part_btn.grid(row=8, column=PARTS_COLUMN, sticky="we")
        self.remove_part_btn = tk.Button(self.window, text="Remove Part", command=self.remove_part)
        self.remove_part_btn.grid(row=9, column=PARTS_COLUMN, sticky="we")
        # bottom menu containing information about the currently selected part
        self.part_types = list(DISTANCE_FUNCTIONS.keys())
        self.part_type_lbl = tk.Label(self.window, text="Type")
        self.part_type_lbl.grid(row=BOTTOM_ROW, column=CANVAS_COLUMN)
        self.part_type_combobox = ttk.Combobox(self.window, state="readonly", values=self.part_types)
        self.part_type_combobox.grid(row=BOTTOM_ROW, column=CANVAS_COLUMN+1, sticky="w")
        self.part_type_combobox.bind("<<ComboboxSelected>>", self.__part_type_selected)
        self.part_args_list = {}
        self.intcmd = self.window.register(self.integer_arg)
        self.floatcmd = self.window.register(self.float_arg)

    def __background_popup(self, event):
        try:
            self.background_menu.tk_popup(event.x_root, event.y_root)
        except:
            pass

    def __load_background(self):
        """
        Ask the user for an image or video to use as a background to better visualize the resulting AOIs.
        """
        video_format_string = "*." + " *.".join(config.VIDEO_FORMATS)
        background_file = filedialog.askopenfilename(initialdir=".",
                                                     title="Open file",
                                                     filetypes=(("Image files","*.png *.jpg *.jpeg *.gif *.tiff *.bmp"),("Video files", video_format_string),("All files","*.*")))
        # cancel on file selection
        if (background_file is None or background_file == "" or background_file == ()):
            return
        if (not os.path.isfile(background_file) or not os.path.exists(background_file)):
            messagebox.showinfo(title = "Not a valid file.", message = f"\'{background_file}\' is not a valid path to a file.")
            return
        _, background_format = os.path.splitext(background_file)
        if (background_format[1:] in config.VIDEO_FORMATS):
            try:
                vid_cap = cv2.VideoCapture(background_file)
            except:
                messagebox.showwarning(title = "Not a valid video.", message = f"The format of \'{background_file}\' is either not supported by opencv or the file may be corrupted.")
                return
            if (not vid_cap.isOpened()):
                messagebox.showwarning(title = "Not a valid video.", message = f"The format of \'{background_file}\' is either not supported by opencv or the file may be corrupted.")
                return
            frame_num = simpledialog.askinteger("Video frame", "Since the provided file is a video a frame needs\nto be specified that is used as background.", parent=self.window, initialvalue=0)
            # cancel on frame selection
            if (frame_num is None):
                return
            if (not vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)):
                messagebox.showwarning(title = "Invalid frame", message = f"Can not get the frame {frame_num} of the given video.")
                return
            success, image = vid_cap.read()
            if (not success):
                messagebox.showwarning(title = "Invalid frame", message = f"Can not read the frame {frame_num} of the given video.")
                return
            # everything should work from here on
            set_background_image(image)
        else:
            image = cv2.imread(background_file)
            if (image is None):
                messagebox.showwarning(title = "Not a valid image.", message = f"The format of \'{background_file}\' is either not supported by opencv or the file may be corrupted.")
                return
            set_background_image(image)
        self.__rescale_background()
        self.redraw()

    def __rescale_background(self):
        canvas_size = (self.draw_canvas.winfo_width(), self.draw_canvas.winfo_height())
        window_size = (self.window.winfo_width(), self.window.winfo_height())
        self.window.geometry(f"{im_size[1] + (window_size[0] - canvas_size[0])}x{im_size[0] + (window_size[1] - canvas_size[1])}")

    def add_grouping(self, name, grouping):
        """
        Add a new grouping.
        """
        self.group_list[name] = grouping
        color = "black"
        for n, a in grouping.items():
            for p in a:
                if (next(filter(lambda x: x >= 68, p[0]), False)):
                    color = "gray"
                    break
            if (color != "black"):
                break
        self.grouping_box.insert(len(self.group_list)-1, name)
        self.grouping_box.itemconfig(len(self.group_list)-1, foreground=color)

    def start(self):
        # starts the window mainloop
        self.window.mainloop()

    def __btn1_pressed(self, event):
        # left mouse button pressed
        self.mouse_pressed = True

    def __btn1_released(self, event):
        # left mouse button released
        self.mouse_pressed = False

    def test_grouping_name(self, name):
        """
        Tests if the given name is a valid, unused name for a grouping.
        """
        if (name is None or name == ""):
            messagebox.showinfo("Empty name", "The name of the grouping may not be empty.")
            return False
        if (name in self.group_list):
            messagebox.showinfo("Name already used", "An existing grouping is already using that name.")
            return False
        if (name != name.strip()):
            messagebox.showinfo("Unvalid name", "That is not a valid name to use.")
            return False
        return True

    def test_aoi_name(self, name):
        """
        Tests if the given name is a valid unused name for an AOI.
        """
        if (name is None or name == ""):
            messagebox.showinfo("Empty name", "The name of the AOI may not be empty.")
            return False
        if (name in self.group_list[self.selected_grouping]):
            messagebox.showinfo("Name already used", "An existing AOI is already using that name.")
            return False
        if (name != name.strip()):
            messagebox.showinfo("Unvalid name", "That is not a valid name to use.")
            return False
        return True

    def __group_selected(self, event):
        # the seleceted grouping changed
        selection = self.grouping_box.curselection()
        if (len(selection) <= 0 or selection[0] < 0):
            self.selected_grouping = None
            return
        self.selected_grouping = self.grouping_box.get(selection[0])
        self.refresh_aoilist()
        self.redraw()

    def __aoi_selected(self, event):
        # the selected AOI changed
        selection = self.aoi_listbox.curselection()
        if (len(selection) <= 0 or selection[0] < 0):
            self.selected_aoi = None
            return
        a_name = self.aoi_listbox.get(selection[0])
        if (self.aoi_sel_index is not None):
            if (self.aoi_sel_index < selection[0]):
                # move behind the currently marked entry
                tmp_name = self.aoi_listbox.get(self.aoi_sel_index)
                tmp = self.group_list[self.selected_grouping][tmp_name]
                new_aoi_dict = {}
                inserted = False
                for n, a in self.group_list[self.selected_grouping].items():
                    if (n == tmp_name):
                        continue
                    new_aoi_dict[n] = a
                    if (n == a_name):
                        inserted = True
                        new_aoi_dict[tmp_name] = tmp
                if (not inserted):
                    print("Insertion of AOI failed!")
                    return
                self.aoi_listbox.delete(self.aoi_sel_index)
                self.aoi_listbox.insert(selection[0], tmp_name)
                self.aoi_sel_index = selection[0]
                self.aoi_listbox.selection_clear(0, tk.END)
                self.aoi_listbox.selection_set(selection[0])
                self.group_list[self.selected_grouping] = new_aoi_dict
                self.redraw()
                return
            elif (self.aoi_sel_index > selection[0]):
                # move before the currently marked entry
                tmp_name = self.aoi_listbox.get(self.aoi_sel_index)
                tmp = self.group_list[self.selected_grouping][tmp_name]
                new_aoi_dict = {}
                inserted = False
                for n, a in self.group_list[self.selected_grouping].items():
                    if (n == tmp_name):
                        continue
                    if (n == a_name):
                        inserted = True
                        new_aoi_dict[tmp_name] = tmp
                    new_aoi_dict[n] = a
                if (not inserted):
                    print("Insertion of AOI failed!")
                    return
                self.aoi_listbox.delete(self.aoi_sel_index)
                self.aoi_listbox.insert(selection[0], tmp_name)
                self.aoi_sel_index = selection[0]
                self.aoi_listbox.selection_clear(0, tk.END)
                self.aoi_listbox.selection_set(selection[0])
                self.group_list[self.selected_grouping] = new_aoi_dict
                self.redraw()
                return
            else:
                return
        self.selected_aoi = a_name
        self.refresh_partslist()
        if (len(self.group_list[self.selected_grouping][self.selected_aoi]) == 1):
            self.parts_listbox.selection_set(0)
            self.selected_part = 0
            self.window.after(100, self.draw_canvas.focus_set)
        self.update_part_info()
        self.redraw()

    def __part_selected(self, event):
        # the selected part of an AOI changed.
        selection = self.parts_listbox.curselection()
        if (len(selection) <= 0 or selection[0] < 0):
            self.selected_part = None
            return
        self.selected_part = selection[0]
        self.update_part_info()
        self.redraw()
        self.window.after(100, self.draw_canvas.focus_set)

    def __part_type_selected(self, event):
        # The type of the selected part was changed.
        selection = self.part_type_combobox.get()
        p_idx = self.selected_part
        if (p_idx is not None):
            part = self.group_list[self.selected_grouping][self.selected_aoi][p_idx]
            args = DISTANCE_FUNCTIONS[part[1]][1]
            vals = []
            for i, arg in enumerate(args.items()):
                # if a value for the parameter is set use it
                if (i < len(part[2])):
                    vals.append(part[2][i])
                else:
                    # use default parameter instead
                    vals.append(arg[1][1])

            self.group_list[self.selected_grouping][self.selected_aoi][p_idx] = (part[0], selection, vals)
            self.refresh_partslist()
            self.parts_listbox.selection_set(p_idx)
            self.selected_part = p_idx
            self.update_part_info()
            self.redraw()

    def __start_aoi_drag(self, event):
        """
        Start of drag and drop in AOI list
        Needed because the order of AOIs is important.
        """
        self.window.after(20, self.__delayed_aoi_drag)

    def __delayed_aoi_drag(self):
        # needed to make drag and drop more smooth.
        idx = self.aoi_listbox.curselection()
        if (idx is None or len(idx) < 1):
            return
        else:
            self.aoi_sel_index = idx[0]
        return

    def __end_aoi_drag(self, event):
        # The drag and drop of AOIs ended.
        self.aoi_sel_index = None

    def __canvas_click(self, event):
        """
        The canvas was clicked. If a grouping, an AOI and a part is selected,
        landmarks are either added or removed from that part.
        """
        self.draw_canvas.focus_set()
        if (self.selected_grouping is None or self.selected_aoi is None or self.selected_part is None):
            return
        width = self.draw_canvas.winfo_width()
        height = self.draw_canvas.winfo_height()
        cur_part = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
        cur_lm = cur_part[0]
        sel_lm = self.selected_landmark
        sel_idx = self.selected_lm_idx
        new_landmarks = [(int(p[0]*width), int(p[1]*height)) for p in SCALED_LANDMARKS]
        closest = aoi.get_closest_point(new_landmarks, (event.x, event.y), 30)
        if (closest is None or closest < 0):
            return
        if (cur_part[1] in UNDIRECTIONAL):
            # unordered landmarks
            if (closest in cur_lm):
                self.selected_landmark = None
                while closest in cur_lm:
                    cur_lm.remove(closest)
            else:
                self.selected_landmark = closest
                self.selected_lm_idx = len(cur_lm) - 1
                cur_lm.append(closest)
        else:
            # order of landmarks is important to the part/ distance function
            if (closest == sel_lm):
                self.selected_landmark = None
                self.selected_lm_idx = None
            else:
                self.selected_landmark = closest
                if (sel_lm is None):
                    if (closest not in cur_lm):
                        self.selected_lm_idx = len(cur_lm)
                        cur_lm.append(closest)
                    else:
                        self.selected_lm_idx = len(cur_lm) - 1 - cur_lm[::-1].index(closest)
                else:
                    if (sel_idx == len(cur_lm) - 1):
                        self.selected_lm_idx = len(cur_lm)
                        cur_lm.append(closest)
                    else:
                        self.selected_lm_idx = sel_idx + 1
                        if (cur_lm[sel_idx + 1] != closest):
                            cur_lm.insert(sel_idx + 1, closest)
        self.redraw()

    def __canvas_key(self, event):
        """
        Delete or backspace was pressed.
        If a landmarks is selected remove it from the currently selected part.
        """
        if (event.keysym == "Delete" or event.keysym == "BackSpace"):
            self.remove_landmark()
            return

    def __window_key(self, event):
        """
        If Escape is pressed the focus is set to the canavas. Mainly used to
        unfocus textboxes and listboxes.
        """
        if (event.keysym == "Escape"):
            self.draw_canvas.focus_set()

    def refresh_canvas(self, event):
        """
        Draw an approximate new canavas and call the draw function.
        """
        scale = (event.width/self.canvas_size[0], event.height/self.canvas_size[1])
        self.draw_canvas.scale("all", 0, 0, scale[0], scale[1])
        self.canvas_size = (event.width, event.height)
        self.window.after(30, self.redraw, event.width, event.height)

    def redraw(self, width=None, height=None):
        """
        If the canvas size is not currently changing, perform a full redraw.
        """
        new_width = self.draw_canvas.winfo_width()
        new_height = self.draw_canvas.winfo_height()
        if (width is not None and height is not None):
            if (new_width != width or new_height != height):
                return
        self.draw_canvas.delete("all")
        #new_landmarks = [(int(p[0]*new_width), int(p[1]*new_height)) for p in SCALED_LANDMARKS]
        if (FACE_IMAGE is not None):
            img = FACE_IMAGE.copy()
        else:
            img = np.full((im_size[0], im_size[1], 3), 255, np.uint8)
        if (self.selected_grouping is not None):
            if (self.selected_grouping in self.group_list):
                grouping = self.group_list[self.selected_grouping]
                for n, a in grouping.items():
                    for p in a:
                        if (next(filter(lambda x: x >= 68, p[0]), False)):
                            self.grouping_box.selection_clear(0, tk.END)
                            self.selected_grouping = None
                            messagebox.showinfo("Background landmarks", "The grouping uses landmarks in the background which are currently not supported by the AOI creator.")
                            self.redraw()
                            return
                visualizations.draw_grouping(img, LANDMARKS, grouping, random_colors=-1, label_aois=False, fill_polygons=True)
            else:
                print(f"Could not find grouping: \'{self.selected_grouping}\'")
        #lm_size = int(min(new_width, new_height)/120)
        lm_size = 3
        sel_part = None
        if (self.selected_grouping is not None and self.selected_aoi is not None):
            if (self.selected_part is not None):
                sel_part = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
                highlight = sel_part[0]
            else:
                highlight = []
                for p in self.group_list[self.selected_grouping][self.selected_aoi]:
                    highlight += p[0]
        else:
            highlight = []

        if (sel_part is not None and sel_part[1] not in UNDIRECTIONAL and len(highlight) > 1):
            p2 = LANDMARKS[highlight[0]]
            for i, idx in enumerate(highlight[1:]):
                p1 = p2
                p2 = LANDMARKS[idx]
                d = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                tl = lm_size*4/d
                cv2.arrowedLine(img, p1, p2, (0, 0, 0), thickness=int(lm_size/3*2), tipLength=tl)

        for i, l in enumerate(LANDMARKS):
            if (i in highlight):
                cv2.circle(img, l, lm_size, (0, 0, 255), -1)
            else:
                cv2.circle(img, l, lm_size, (0, 0, 0), -1)

        if (self.selected_landmark is not None):
            cv2.circle(img, LANDMARKS[self.selected_landmark], int(lm_size*1.5), (0, 255, 0), -1)

        if (FACE_IMAGE is not None):
            img = cv2.addWeighted(FACE_IMAGE, 0.3, img, 0.7, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (new_width, new_height))
        self.aoi_image = Image.fromarray(img, "RGB")
        self.canvas_image = ImageTk.PhotoImage(image=self.aoi_image)
        self.draw_canvas.create_image(0, 0, anchor="nw", image=self.canvas_image)

    def new_grouping(self):
        """
        The user wants to create a new grouping.
        """
        name_ok = False
        while not name_ok:
            name = simpledialog.askstring("Grouping name", "Type in a name for the new grouping.", parent=self.window)
            if (name is None):
                return
            name_ok = self.test_grouping_name(name)
        self.grouping_box.insert(len(self.group_list), name)
        self.group_list[name] = {}

    def remove_grouping(self):
        """
        The user wants to remove the currently selected grouping.
        """
        g_name = self.selected_grouping
        selection = self.grouping_box.curselection()
        if (selection is None or len(selection) <= 0):
            messagebox.showinfo("No grouping selected", "No grouping to rename was selected.")
            return
        answer = messagebox.askyesno("Delete grouping?", f"Are you sure you want to delete {g_name}")
        if (answer):
            del self.group_list[g_name]
            self.selected_grouping = None
            self.grouping_box.delete(selection[0])
        else:
            return

    def rename_grouping(self):
        """
        The user wants to rename the currently seleceted grouping.
        """
        old_name = self.selected_grouping
        selection = self.grouping_box.curselection()
        if (len(selection) <= 0 or selection[0] < 0):
            messagebox.showinfo("No grouping selected", "No grouping to rename was selected.")
            return
        name_ok = False
        while not name_ok:
            name = simpledialog.askstring("Grouping name",
                                          "Type in a new name for the grouping.",
                                          parent=self.window,
                                          initialvalue=old_name)
            if (name is None):
                return
            if (name == old_name):
                return
            name_ok = self.test_grouping_name(name)
        self.group_list = {name if k == old_name else k: v for k, v in self.group_list.items()}
        self.grouping_box.delete(selection[0])
        self.grouping_box.insert(selection[0], name)
        self.grouping_box.selection_set(selection[0])
        self.selected_grouping = name

    def new_aoi(self):
        """
        The user wants to create a new AOI.
        """
        g_name = self.selected_grouping
        name_ok = False
        while not name_ok:
            name = simpledialog.askstring("AOI name", "Type in a name for the new AOI.", parent=self.window)
            if (name is None):
                return
            name_ok = self.test_aoi_name(name)
        self.aoi_listbox.insert(len(self.group_list[g_name]), name)
        self.group_list[g_name][name] = []

    def remove_aoi(self):
        """
        The user wants to remove the currently seleceted AOI.
        """
        a_name = self.selected_aoi
        g_name = self.selected_grouping
        selection = self.aoi_listbox.curselection()
        if (selection is None or len(selection) <= 0):
            messagebox.showinfo("No AOI selected", "No AOI to remove was selected.")
            return
        answer = messagebox.askyesno("Delete AOI?", f"Are you sure you want to delete {a_name}")
        if (answer):
            self.group_list[g_name].pop(a_name)
            self.selected_aoi = None
            self.aoi_listbox.delete(selection[0])
            self.redraw()
        else:
            return

    def rename_aoi(self):
        """
        The user wants to rename the currently seleceted AOI.
        """
        old_name = self.selected_aoi
        selection = self.aoi_listbox.curselection()
        if (len(selection) <= 0 or selection[0] < 0):
            messagebox.showinfo("No AOI selected", "No AOI to rename was selected.")
            return
        name_ok = False
        while not name_ok:
            name = simpledialog.askstring("AOI name",
                                          "Type in a new name for the AOI.",
                                          parent=self.window,
                                          initialvalue=old_name)
            if (name is None):
                return
            if (name == old_name):
                return
            name_ok = self.test_aoi_name(name)
        self.group_list[self.selected_grouping] = {name if k == old_name else k: v for k, v in self.group_list[self.selected_grouping].items()}
        self.aoi_listbox.delete(selection[0])
        self.aoi_listbox.insert(selection[0], name)
        self.aoi_listbox.selection_set(selection[0])
        self.selected_aoi = name

    def new_part(self):
        """
        The user wants to create a new part for the currently selected AOI.
        """
        g_name = self.selected_grouping
        a_name = self.selected_aoi
        if (g_name is None or a_name is None):
            messagebox.showinfo("No AOI selected", "Please select an AOI to which the new part should be added.")
        dist_func = next(k for k in DISTANCE_FUNCTIONS)
        self.parts_listbox.insert(tk.END, dist_func)
        self.group_list[g_name][a_name].append(([], dist_func, []))
        self.parts_listbox.selection_set(tk.END)
        self.selected_part = len(self.group_list[g_name][a_name])-1
        self.update_part_info()

    def remove_part(self):
        """
        the user wants to remove the currently selected part.
        """
        a_name = self.selected_aoi
        g_name = self.selected_grouping
        selection = self.parts_listbox.curselection()
        if (selection is None or len(selection) <= 0):
            messagebox.showinfo("No Part selected", "No Part to remove was selected.")
            return
        answer = messagebox.askyesno("Delete Part?", f"Are you sure you want to delete {self.parts_listbox.get(selection[0])}")
        if (answer):
            self.group_list[g_name][a_name].pop(selection[0])
            self.selected_part = None
            self.parts_listbox.delete(selection[0])
            self.update_part_info()
            self.redraw()
        else:
            return

    def remove_landmark(self):
        """
        Remove te currently selected landmarks from the selected part.
        """
        if (self.selected_grouping is None or self.selected_aoi is None or self.selected_part is None or self.selected_landmark is None or self.selected_lm_idx is None):
            return
        cur_part = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
        cur_lm = cur_part[0]
        sel_idx = self.selected_lm_idx
        if (cur_part[1] in UNDIRECTIONAL):
            while self.selected_landmark in cur_lm:
                cur_lm.remove(self.selected_landmark)
            self.selected_landmark = None
            self.selected_lm_idx = None
        else:
            del cur_lm[sel_idx]
            if (sel_idx > 0):
                self.selected_lm_idx = sel_idx - 1
                self.selected_landmark = cur_lm[self.selected_lm_idx]
            elif (len(cur_lm) > 0):
                self.selected_landmark = cur_lm[sel_idx]
            else:
                self.selected_landmark = None
                self.selected_lm_idx = None
        self.redraw()

    def save_groupings(self):
        """
        The user wants to save the groupings to a config file.
        """
        file_name = filedialog.asksaveasfilename(
            parent=self.window,
            title="Select config file to save",
            initialdir=".",
            initialfile="test_config.py",
            filetypes=[("Python files", "*.py"), ("All files", "*")]
        )
        if (file_name is None or len(file_name) <= 0):
            return
        success = write_groupings(self.group_list, file_name)
        if (success):
            messagebox.showinfo("Saved groupings", f"Saved groupings to ’{file_name}’.")
            print("Saved")
        else:
            messagebox.showwarning("Saving failed!", f"Could not save groupings to '{file_name}'. Make sure the file is a valid config file ore contains a dictionary named \'AOI_GROUPINGS\'")

    def refresh_aoilist(self):
        """
        Refresh the list containing all AOis of the currently selected grouping.
        """
        self.selected_aoi = None
        self.aoi_listbox.delete(0, tk.END)
        if (self.selected_grouping is not None):
            if (self.selected_grouping not in self.group_list):
                print("Grouping list desynced.")
                self.refresh_partslist()
                return
            for an, a in self.group_list[self.selected_grouping].items():
                self.aoi_listbox.insert(tk.END, an)
        self.refresh_partslist()

    def refresh_partslist(self):
        """
        Refresh the list containing all parts of the currently selected AOI.
        """
        self.selected_part = None
        self.parts_listbox.delete(0, tk.END)
        if (self.selected_aoi is not None):
            if (self.selected_grouping not in self.group_list):
                print(self.selected_grouping)
                print(self.selected_aoi)
                self.update_part_info()
                print("Grouping list desynced.")
                return
            if (self.selected_aoi not in self.group_list[self.selected_grouping]):
                print("AOI list desynced.")
                self.update_part_info()
                return
            for p in self.group_list[self.selected_grouping][self.selected_aoi]:
                self.parts_listbox.insert(tk.END, p[1])
        self.update_part_info()

    def integer_arg(self, value, index):
        """
        Creates an input for an integer.
        """
        p = None
        if (self.selected_grouping is not None and self.selected_aoi is not None and self.selected_part is not None):
            p = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
            if (len(p[2]) <= int(index)):
                extend = int(index)-len(p[2])+1
                args = p[2] + [0]*extend
                p = (p[0], p[1], args)
        if (value):
            try:
                v = int(value)
                if (p is not None):
                    p[2][int(index)] = v
                    self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part] = p
                    self.redraw()
                return True
            except ValueError:
                return value == ""
        else:
            if (p is not None):
                p[2][int(index)] = 0
                self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part] = p
                self.redraw()
            return True

    def float_arg(self, value, index):
        """
        Creates an input for a float value.
        """
        p = None
        if (self.selected_grouping is not None and self.selected_aoi is not None and self.selected_part is not None):
            p = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
            if (len(p[2]) <= int(index)):
                extend = int(index)-len(p[2])+1
                args = p[2] + [0]*extend
                p = (p[0], p[1], args)
        if (value):
            try:
                v = float(value)
                if (p is not None):
                    p[2][int(index)] = v
                    self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part] = p
                    self.redraw()
                return True
            except ValueError:
                return value == ""
        else:
            if (p is not None):
                p[2][int(index)] = 0
                self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part] = p
                self.redraw()
            return True

    def bool_arg(self, index):
        """
        Creates a checkbox to input a boolean value.
        """
        if (self.selected_grouping is not None and self.selected_aoi is not None and self.selected_part is not None):
            p = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
            if (len(p[2]) <= int(index)):
                extend = int(index)-len(p[2])+1
                args = p[2] + [0]*extend
                p = (p[0], p[1], args)
            p[2][index] = not p[2][index]
            self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part] = p
            self.redraw()

    def update_part_info(self):
        """
        Updates the information regarding the selected part.
        This section can than be used to for example change the thickness of the line.
        """
        for name, t in self.part_args_list.items():
            for widget in t:
                if (widget is not None):
                    widget.destroy()
        self.selected_landmark = None
        self.selected_lm_idx = None
        if (self.selected_part is not None):
            part = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part]
            self.part_type_combobox.set(part[1])
            args = DISTANCE_FUNCTIONS[part[1]][1]
            arg_names = list(args.keys())
            d_offset = 0
            self.part_args_list = {}
            for arg_idx, arg_name in enumerate(args.keys()):
                # get current set value or default if not set
                current = part[2][arg_idx] if arg_idx < len(part[2]) else args[arg_name][1]
                # create a label for the parameter
                lbl = tk.Label(self.window, text=arg_name)
                lbl.grid(row=BOTTOM_ROW+1+arg_idx, column=CANVAS_COLUMN)
                if (args[arg_name][0] == bool):
                    val = tk.Checkbutton(self.window, command=lambda: self.bool_arg(arg_idx))
                    if (current):
                        val.select()
                    val.grid(row=BOTTOM_ROW+1+arg_idx, column=CANVAS_COLUMN+1, sticky="w")
                    self.part_args_list[arg_name] = (lbl, val)
                elif (args[arg_name][0] == int):
                    val = tk.Entry(self.window, validate="key", validatecommand=(self.intcmd, "%P" , str(arg_idx)))
                    val.insert(0, current)
                    val.grid(row=BOTTOM_ROW+1+arg_idx, column=CANVAS_COLUMN+1, sticky="w")
                    self.part_args_list[arg_name] = (lbl, val)
                elif (args[arg_name][0] == float):
                    val = tk.Entry(self.window, validate="key", validatecommand=(self.floatcmd, "%P" , str(arg_idx)))
                    val.insert(0, current)
                    val.grid(row=BOTTOM_ROW+1+arg_idx, column=CANVAS_COLUMN+1, sticky="w")
                    self.part_args_list[arg_name] = (lbl, val)
                else:
                    self.part_args_list[arg_name] = (lbl, None)
            if (len(self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part][0]) > 0):
                self.selected_landmark = self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part][0][-1]
                self.selected_lm_idx = len(self.group_list[self.selected_grouping][self.selected_aoi][self.selected_part][0]) - 1
        else:
            self.part_type_combobox.set("")

# create and setup window then start it
win = AOI_editor()
for n, g in AOI_GROUPINGS.items():
    win.add_grouping(n, g)
win.start()

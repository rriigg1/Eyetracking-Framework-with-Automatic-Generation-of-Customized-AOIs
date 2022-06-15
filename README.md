# Automatic Generation of Customized AOIs and Evaluation of Observers' Gaze in Portrait Videos
If you use this in your research please cite:
```
@inproceedings{wohler2022automatic,
  title = {Automatic Generation of Customized {AOI}s and Evaluation of Observers' Gaze in Portrait Videos},
  author = {W{\"o}hler, Leslie and von Estorff, Moritz and Castillo, Susana  and Magnor, Marcus},
  booktitle = {Proceedings of the {ACM} on Human-Computer Interaction },
  doi = {10.1145/3530885},
  volume = {6},
  number = {{ETRA}},
  pages = {144:1--144:14},
  year = {2022}
}
```
The paper can be found at: [https://graphics.tu-bs.de](https://graphics.tu-bs.de/publications/wohler2022automatic)

### Needed python packages:
* [xlrd](https://xlrd.readthedocs.io/en/latest/): `pip install xlrd`
* [openpyxl](https://openpyxl.readthedocs.io/en/stable/): `pip install openpyxl`
* [NumPy](https://numpy.org/): `pip install numpy`
* [Dlib](http://dlib.net/): `pip install dlib`
* [OpenCV](https://github.com/skvark/opencv-python): `pip install opencv-python`
* [EditDistance](https://github.com/roy-ht/editdistance): `pip install editdistance`
* [SciPy](https://www.scipy.org/): `pip install scipy`
* [tqdm](https://tqdm.github.io/): `pip install tqdm`
* Tkinter is used for a simple GUI to create and edit AOIs
    * Tkinter comes with Python when using Windows

Or use `pip install -r requirements.txt` to install all requirements besides Tkinter

#### Dlib shape prediction model
A trained Dlib shape predictor model is needed to produce the landmarks.
Such a model can be found at: [dlib.net](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### Usage:
The framework can either be imported as a package directly:
```py
import EyetrackingFramework as ETF
```
or be used via the commandline.
For this a launcher.py is provided which acts as a mainfile and takes input as parameters.
The various methods of the framework can be accessed this way.
Different methods can be chained together so that the output from a prior method is used as input for a later method.
By default the output of the last method is saved to a default directory. To prevent this **end** can be attached to the command.

To edit the existing AOIs that can be used for analysis or to create new AOIs use the command:
```sh
python aoi_creator.py
```

### Functions:

Preprocess data:
```sh
python launcher.py preprocess data/*
```

Generate landmarks for videos:
```sh
python launcher.py landmarks videos/*
```

With given fixation data get the aois fixated by the participant using a specific grouping from *config.py*:
```sh
python launcher.py preprocess data/* load landmarks/* get-aoi -p FACE_SIMPLE
```

Count the fixations per AOI saccade counts work similar:
```sh
python launcher.py load aoi_fixations/* count-aoi-fixations
```

Calculate the Levenshtein distance between a given scanpath and the rest of the data:
```sh
python launcher.py load aoi_fixations/* levenshtein <video> <participant number>
```

Calculate the Mannan distance between a given scanpath and the rest of the data:
```sh
python launcher.py preprocess data/* mannan <video> <participant number>
```

To load data of type preprocessed data, landmarks or aois:
```sh
python launcher.py load data.csv
```
or to load multiple files:
```sh
python launcher.py load data/*
```

Visualizations of different kinds as seen in the paper:
```sh
python launcher.py visualize AOIS <image>
python launcher.py load landmarks/* visualize AOIS <video>
python launcher.py load landmarks/* load fixations/* visualize SCANPATH <image/video>
python launcher.py load landmarks/* load fixations/* heatmap heatmap.png -i <background image>
python launcher.py load landmarks/* load fixations/* heatmap heatmap.png -i <background video> -f <frame number>
```



### Examples:
Given:
Fixation data consisting of one file in directory *example_fixations*
Videos in directory *example_videos*

#### Landmarks
It is best to generate the landmarks for all needed videos beforehand and save them since this takes a while and saves time if you need them later on.
```sh
python launcher.py landmarks example_videos/*
```
To save them to a specific directory:
```sh
python launcher.py landmarks example_videos/* save <directory>
```

#### AOI fixations
Next step is to generate the lists of fixated AOIs:
```sh
python launcher.py preprocess example_fixations/* load landmarks/* get-aoi -p FACE_SIMPLE
```
To save them to a specific directory:
```sh
python launcher.py preprocess example_fixations/* load landmarks/* get-aoi -p FACE_SIMPLE save <directory>
```
To test if it works as exspected:
```sh
python launcher.py preprocess example_fixations/* load landmarks/* get-aoi -v -p FACE_SIMPLE
```

#### Statistics and analysis
To do fixation counts:
```sh
python launcher.py load aoi_fixations/* count-aoi-fixations
```

To do Levenshtein distance (edit distance based on fixated aois)
In this case compare scanpath of *video1* from participant *2* with the rest of the data:
```sh
python launcher.py load aoi_fixations/* levenshtein video1 2
```
or to compare with all data that contains a pattern in the name:
```sh
python launcher.py load aoi_fixations/*<pattern>* aoi_fixations/*video1_2* levenshtein video1 2
```

To do Mannan distance
In this case compare scanpath of *video1* from participant *2* with the rest of the data:
```sh
python launcher.py preprocess example_reports/* mannan video1 2
```

If wanted this can also be done in one command:
```sh
python launcher.py preprocess example_fixations/* landmarks example_videos/* get-aoi -p FACE_SIMPLE count-aoi-fixations save output levenshtein video1 2 save output
```

To generate a file comparing the results of different groupings:
```sh
python launcher.py preprocess example_reports/* load landmarks/* get-aoi -p FACE_SIMPLE get-aoi -p FACE_SIMPLE_POINTS save aoi_fixations count-aoi-fixations save output
```

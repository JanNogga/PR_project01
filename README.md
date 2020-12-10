# PR_project01
Solutions to the Pattern Recognition lecture first group project.

[![forthebadge](https://forthebadge.com/images/badges/built-with-resentment.svg)](https://forthebadge.com)

The code assumes that this repository is placed into the same folder as the *LecturePatternRecognition* repository. For example, your folder structure could look like this:

```bash
.
└── SomeParentDirectory
    ├── LecturePatternRecognition
    │   ├── README.md
    │   ├── project01
    │   │   ├── android.csv
    │   │   └── project01.pdf
    │   ├── sheet01
    │   │   ├── patternrec.yml
    │   │   └── sheet01.pdf
    │   └── sheet02
    │       ├── data.csv
    │       └── sheet02.pdf
    └── PR_project01
        ├── README.md
        └── utils.py
```

Please create your own branches from the main repository branch and work in those, we can discuss all necessary merges together in meetings. For me, creating a new branch looked like this:

```bash
lnogga@DESKTOP-QP6646L:/mnt/c/Users/Jan/sciebo/WS2021/Pattern_Recognition/PR_project01$ git checkout -b Jan
Switched to a new branch 'Jan'

lnogga@DESKTOP-QP6646L:/mnt/c/Users/Jan/sciebo/WS2021/Pattern_Recognition/PR_project01$ git push --set-upstream origin Jan
Total 0 (delta 0), reused 0 (delta 0)
remote:
remote: Create a pull request for 'Jan' on GitHub by visiting:
remote:      https://github.com/JanNogga/PR_project01/pull/new/Jan
remote:
To https://github.com/JanNogga/PR_project01.git
 * [new branch]      Jan -> Jan
Branch 'Jan' set up to track remote branch 'Jan' from 'origin'.
```
To run the current code, you need to install *pandas*, a Python Data Analysis library that makes it very easy to work with tabular data. To install this, activate your pattern recognition conda environment

```bash
lnogga@DESKTOP-QP6646L:/mnt/c/Users/Jan$ conda activate patternrec
```

and install the corresponding conda package

```bash
(patternrec) lnogga@DESKTOP-QP6646L:/mnt/c/Users/Jan$ conda install pandas
```

While you are at it, I also recommend installing *seaborn*, a statistical data visualization library. It synergizes with *pandas*, allowing insight into the data we have been given with very simple commands. Run 

```bash
(patternrec) lnogga@DESKTOP-QP6646L:/mnt/c/Users/Jan$ conda install seaborn
```

to install this conda package, too. 

Currently, there isn't much code, but what is there is in *utils.py*. I suggest you import the functions included there, for example, into your jupyter notebooks:

```python
from utils import get_table

android_data = get_table()
android_data.head(10)
```

to print the first 10 rows of the table. If you include a docstring for your functions like in the definition of *get_table()*, others can call *help()* to get a quick rundown of what your function does, for example

```python
help(get_table)
```

prints

```
Help on function get_table in module utils:

get_table(filepath=None)
    Returns android.csv as a DataFrame
    
    This function assumes that code is in a directory which has the same parent directory as the LecturePatternRecognion
    repository. If this is not the case, please supply a string with the correct file path to the filepath argument.
```

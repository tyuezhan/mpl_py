# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["mpl", "mpl_utils"],
    package_dir={"mpl": "src/mpl", "mpl_utils": "src/mpl_utils"},
)

setup(**setup_args)


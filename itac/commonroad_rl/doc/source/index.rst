.. CommonRoad_RL documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
CommonRoad-RL
=============

This project contains a software package to solve motion planning problems on CommonRoad
using reinforcement learning methods, currently based on `OpenAI Stable Baselines <https://stable-baselines.readthedocs.io/en/master/>`__.

The software is written in Python 3.7 and tested on Linux 18.04. The usage of the Anaconda_ Python distribution is strongly recommended.

.. _Anaconda: http://www.anaconda.com/download/#download


.. seealso::
	
	* `CommonRoad <https://commonroad.in.tum.de/>`__
	* `CommonRoad-io <https://commonroad.in.tum.de/commonroad_io>`__
	* `CommonRoad Drivability Checker <https://commonroad.in.tum.de/drivability_checker>`__
	* `Vehicle Models <https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/tree/master>`__
	* `Spinning Up <https://spinningup.openai.com/en/latest/>`__
	* `OpenAI Gym <https://gym.openai.com/docs/>`__
	* `OpenAI Safety Gym <https://openai.com/blog/safety-gym/>`__


Prerequisits
============

This project should be run with `conda <https://www.anaconda.com/>`__. Make sure it is installed before proceeding with the installation. Initialize conda::
   
   /path/to/conda/bin/conda init

Install build packages::

   sudo apt-get update
   sudo apt-get install build-essential make cmake


Installation
============

Currently, the package can only be installed from the repository. First, clone it::

	git clone https://gitlab.lrz.de/tum-cps/commonroad-rl.git

To create an environment for this project including all requirements, run::

   conda env create -n cr37 -f environment.yml

After the repository is cloned, CommonRoad-RL can be installed without sudo rights with::

	bash scripts/install.sh -e cr37 --no-root

and with sudo rights::

	bash scripts/install.sh -e cr37

:code:`cr37` to be replaced by the name of your conda environment if needed.

This will build all softwares in your home folder. You can press ``ctrl`` + ``c`` to skip when asked for sudo password.
Note that all necessary libraries need to be installed with sudo rights beforehands.
Please ask your admin to install them for you if they are missing.

(optional) Install pip packages for the docs. If you want to use jupyter notebook for the tutorials, also install jupyter::

   source activate cr37
   pip install -r commonroad_rl/doc/requirements_doc.txt
   conda install jupyter


Test if installation succeeds
============
Further details of our test system refer to ``./commonroad_rl/tests``. Run tests::

   source activate cr37
   pytest commonroad_rl/tests --scope unit module -m "not slow"

Changelog
============

 

Getting Started
===============

A tutorial on the main functionalities can be found in the form of jupyter notebooks in the :code:`tutorials` folder.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   module/index.rst


..Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

Contact information
===================

:Website: `http://commonroad.in.tum.de <https://commonroad.in.tum.de/>`_
:Email: `commonroad@lists.lrz.de <commonroad@lists.lrz.de>`_

.. _tests-index:

=====
Tests
=====

**CommonRoad-RL Testing**

This module contains the test system of the reposiroty
`CommonRoad-RL <https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl>`__.
It uses `pytest <https://docs.pytest.org/en/stable/>`__ for test
management with customized markers.

**Test structure**

::

    └── tests
        ├── common
        |   └── marker.py                   -> Test marking helper methods
        ├── external
        |   └── test_<external_module_1>.py -> Contains tests relevant to <external_module_1>
        ├── resources                       -> Resources for tests
        |   └── <test_name>  
        |       └── <resource_file>                 
        ├── references                      -> Contains reference files for the tests
        |   └── <test_name>     
        |       └── <reference_file>
        ├── outputs                         -> Contains the outputs of the last test run
        |   └── <test_name>     
        |       └── <output_file>
        ├── <module_1>                      -> Contains tests relevant to <module_1>
        |   ├── <submodule_1_1>                  -> Contains unittests of <module_1>.<submodule_1_1>
        |   |   └── test_<script>.py        -> Contains UNIT tests of <script>
        |   └── test_<module_1>.py          -> Contains MODULE tests of <module_1>
        ├── <module_2>
        :
        ├── <module_N>
        ├── test_commonroad_rl              -> Contains INTEGRATION tests
        ├── conftest.py                     -> Test configuration script
        ├── pytest.ini                      -> Test configuration settings
        └── README.md                       -> This file

**Test configuration**

For the better test organization attributes can be defined using
`markers <https://docs.pytest.org/en/latest/example/markers.html>`__.
Using these attributes the tests can be filtered. To make the indent
clean and help to organize the tests, two mandatory markers has been
defined for the tests: \* RunScope \* unit \* module \* integration \*
RunType \* functional \* non\_functional

The definition of this types can be found under
`common/marker.py </testsmon/marker.py>`__. **These two markers needs to
be defined for all tests.** The usage of custom markers has been
restricted to make it easier to see what kind of markers are in use.
Further markers: \* slow \* serial

If further markers are needed, define it in the
`pytest.ini </testsest.ini>`__ file and add a wrapper in
`common/marker.py </testsmon/marker.py>`__.

* Advanced configuration

If you need advanced test configuration you can use the
`conftest.py </testsftest.py>`__ and define
`hooks <https://docs.pytest.org/en/latest/writing_plugins.html#writing-hook-functions>`__.

* RunScope

  *-Unit*


  A unit test checks the behavior of the smallest units of the program. It should cover only 
  a **single function or a class**. These tests must be as independent as possible from the 
  other units.

  *-Module*


  A module test checks the interaction between the units. It should cover bigger functions, 
  which **uses more units of the given module**, but it can't depend on other internal module.

  *-Integration*


  The integration test checks the behavior of the whole repository. It can use any internal modules.

* RunType

  *-Functional*


  A functional test checks the input-output mapping of the program.

  *-Non-functional*


  A non-functional test checks the non-functional aspects, which can be performance 
  (time consumption, CPU usage, memory usage), reliability, repeatability, etc. In most 
  of the cases the expected value is an interval.

**Write test**

The pytest framework discovers all methods in the form
``test_<method>``. Therefore you have to define your test function with
a preceding ``test_`` string.

**For all tests you have to define the two mandatory markers**. For
better maintainability, please use the wrappers in the
`common/marker.py </testsmon/marker.py>`__.

.. code:: python

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [(np.pi * 2, 0),
         (-3 / 2 * np.pi, np.pi / 2),
         (7 / 2 * np.pi, -np.pi / 2),
         (3 / 2 * np.pi, -np.pi / 2)])
    @unit_test
    @functional
    def test_shift_orientation(angle, expected):
        shifted_angle = shift_orientation(angle)
        assert np.isclose(shifted_angle, expected)

You can use other features of pytest as well.

* Use resource files

If your test requires resource filef, you should place them under the
``test/resources/<test_method_name>`` folder. To use it, please use the
function provided in the ``tests/common/path.py`` script

.. code:: python

    import os
    from tests.common.path import resource_root

    resource_file_path = os.path.join(resource_root("<test_name>"), "<test_file>")

**Note: If you use resource files, always make sure that the files are
not ignored and pushed to the server.**

* Use reference files

If your test creates files which should be compared against given
reference files, you should place the reference files in the
``tests/references/<test_method_name>`` folder. The output files of your
test should be placed in the ``tests/outputs/<test_method_name>``
folder. This folder is automatically cleared before the test run.

.. code:: python

    import os
    from tests.common.path import reference_root, output_root

    output_file_path = os.path.join(output_root("<test_name>"), "<output_file>")
    reference_file_path = os.path.join(reference_root("<test_name>"), "<output_file>")
    function_creates_file_to(output_file_path)
    assert files_equals(output_file_path, reference_file_path)

**Note: If you use reference files, always make sure that the files are
not ignored and pushed to the server.**

**Run test**

You have to call the test from the root folder of the repository ### Run
a single test You can run a specific test as follows

.. code:: bash

    pytest <module>/[unittests]/test_<script>.py::test_<method>

* Run a batch of tests

For the test runs you can define the two main filters using \* --scope
... \* --type ...

Example:

.. code:: bash

    pytest commonroad_rl/tests --scope unit module --type functional -m "not slow"

By default the tests are not filtered, all tests will be run.

For more and detailed options, please refer to
`pytest <https://docs.pytest.org/en/stable/usage.html>`__

Common
======

Evaluation
----------
.. automodule:: commonroad_rl.tests.common.evaluation
   :members:
   :member-order: bysource

Marker
------
.. automodule:: commonroad_rl.tests.common.marker
   :members:
   :member-order: bysource

``RunScope`` class
^^^^^^^^^^^^^^^^^^
.. autoclass:: RunScope
   :members:
   :member-order: bysource

``RunType`` class
^^^^^^^^^^^^^^^^^
.. autoclass:: RunType
   :members:
   :member-order: bysource

Non Functional
--------------
.. automodule:: commonroad_rl.tests.common.non_functional
   :members:
   :undoc-members:
   :member-order: bysource

Path
----
.. automodule:: commonroad_rl.tests.common.path
   :members:
   :member-order: bysource

Gym CommonRoad
==============

Feature Extraction
------------------

Test Goal
^^^^^^^^^
.. automodule:: commonroad_rl.tests.gym_commonroad.feature_extraction.test_goal
   :members:
   :undoc-members:
   :member-order: bysource

Test Surroundings
^^^^^^^^^^^^^^^^^
.. automodule:: commonroad_rl.tests.gym_commonroad.feature_extraction.test_surroundings
   :members:
   :undoc-members:
   :member-order: bysource

Utils
-----

Test Scenario
^^^^^^^^^^^^^
.. automodule:: commonroad_rl.tests.gym_commonroad.utils.test_scenario
   :members:
   :undoc-members:
   :member-order: bysource

Test Action
-----------
.. automodule:: commonroad_rl.tests.gym_commonroad.test_action
   :members:
   :undoc-members:
   :member-order: bysource

Test CommonRoad Env
-------------------
.. automodule:: commonroad_rl.tests.gym_commonroad.test_commonroad_env
   :members:
   :undoc-members:
   :member-order: bysource

Test Gym CommonRoad
-------------------
.. automodule:: commonroad_rl.tests.gym_commonroad.test_gym_commonroad
   :members:
   :undoc-members:
   :member-order: bysource

Test Observation
----------------
.. automodule:: commonroad_rl.tests.gym_commonroad.test_vehicle
   :members:
   :undoc-members:
   :member-order: bysource

Test Vehicle
------------
.. automodule:: commonroad_rl.tests.gym_commonroad.test_vehicle
   :members:
   :undoc-members:
   :member-order: bysource

Utils Run
=========

Utils
-----

Test Hyperparams Opt
^^^^^^^^^^^^^^^^^^^^
.. automodule:: commonroad_rl.tests.utils_run.utils.test_hyperparams_opt
   :members:
   :undoc-members:
   :member-order: bysource

Test Observation Configs Opt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: commonroad_rl.tests.utils_run.utils.test_observation_configs_opt
   :members:
   :undoc-members:
   :member-order: bysource

Test Reward Configs Opt
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: commonroad_rl.tests.utils_run.utils.test_reward_configs_opt
   :members:
   :undoc-members:
   :member-order: bysource

Config Test
===========
.. automodule:: commonroad_rl.tests.conftest
   :members:
   :member-order: bysource

Test Commonroad RL
==================
.. automodule:: commonroad_rl.tests.test_commonroad_rl
   :members:
   :undoc-members:
   :member-order: bysource


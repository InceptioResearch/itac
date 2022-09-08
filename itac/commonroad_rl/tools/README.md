# Tools for the Commonroad RL package

Short descriptions for all tools and detailed usage hints can usually be found in each script
and would be merely redundant here.
Usually, a script in this folder is run by
```shell
python -m commonroad_rl.tools.SCRIPT_NAME
```

### Modification of the CR XSD specification

A modified xml specification is supplied together with this package.
The main reason is that for the datasets that are drone recorded, vehicles are bound to "spawn" at the boundary of the scenario.
This is undesired behaviour in the commonroad framework and hence disallowed by the xml specification (initial states time_step is forced to zero).
For this project this does not bother.
A slightly modified .xsd file has thus been created to allow for non-negative time_step values in the initial state of objects (rather than only zero).

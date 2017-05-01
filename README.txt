How to run:

Requires Python 2.7 with Numpy

1) Download and build "kristofe"'s fork of MantaFlow: https://github.com/kristofe/manta 
		Remember to build with the flag "-DGUI=ON".
		Build instructions can be found at MantaFlow's official page: http://mantaflow.com/install.html
2) Place "r1turb.py" in the folder "/manta/scenes/"
3) Create a folder "voxels" at the same folder level as MantaFlows main folder "manta".
4) Place "R1_64.binvox" in the folder "voxels"
4) cd into "/manta/build" and run "./manta ../scenes/r1turb.py"
5) When running the simulation, particles can be shown or hidden with the shortcut "Alt-b" which
		is useful to reveal the velocity field. Further shortcuts can be found by clicking the 
		button marked "?". 

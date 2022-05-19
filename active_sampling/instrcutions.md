#Instructions

1. First install pcg-gazebo from https://github.com/boschresearch/pcg_gazebo
2. Write a yaml file describing the gazebo worls
3. To generate file ensure yaml file hgas custom models imported as separate assets
4. Run the ./gen_model.sh WORLD_CONFIG_FILE to create a world.sdf and gazebo sim to test world
5. go /{HOME}/.gazebo/worlds to find your world folder and corresponding sdf
6. To generate interest array, ensure yaml file ahas all assets defined under one assets tag as shown in randy.yml
7. Run the gazebo_onfig script with your WORLD_CONFIG_FILE
